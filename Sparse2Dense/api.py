from __future__ import annotations

"""
English:
Public API for Sparse2Dense. This module exposes the high-level project runner.

Portuguese:
API publica do Sparse2Dense. Este modulo expoe o executor de projeto em alto nivel.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .calibration import CameraLidarCalibration, load_calibration
from .loaders import load_pointcloud
from .pipeline import (
    densify_projection,
    load_image,
    make_debug_image,
    prepare_projection,
    save_depth_png_mm,
    save_intensity_png,
)
from .torch_backend import (
    densify_projection_torch,
    prepare_projection_torch,
    torch_cuda_synchronize,
    torch_cuda_available,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
POINTCLOUD_EXTENSIONS = {".bin", ".ply", ".pcd", ".txt", ".csv"}


@dataclass(frozen=True)
class InputPair:
    """
    English:
    One matched image/point-cloud pair to be processed.

    Portuguese:
    Um par imagem/point-cloud casado para processamento.
    """

    frame_id: str
    image_path: Path
    pointcloud_path: Path


@dataclass
class ProjectSummary:
    """
    English:
    Final execution report returned by the framework.

    Portuguese:
    Relatorio final de execucao retornado pelo framework.
    """

    output_root: str
    device_used: str
    total_pairs_found: int
    depth_generated: int
    intensity_generated: int
    debug_generated: int
    unmatched_images: list[str]
    unmatched_pointclouds: list[str]
    shared_compute_reused: bool
    total_wall_time_s: float
    depth_total_time_s: float
    depth_average_time_s: float
    intensity_total_time_s: float
    intensity_average_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "Sparse2Dense summary",
            f"Output root: {self.output_root}",
            f"Device used: {self.device_used}",
            f"Pairs found: {self.total_pairs_found}",
            f"Depth maps generated: {self.depth_generated}",
            f"Intensity maps generated: {self.intensity_generated}",
            f"Debug images generated: {self.debug_generated}",
            f"Total wall time: {self.total_wall_time_s:.3f}s",
        ]
        lines.append(
            f"Depth total time: {self.depth_total_time_s:.3f}s | "
            f"Depth average time: {self.depth_average_time_s:.3f}s"
        )
        lines.append(
            f"Intensity total time: {self.intensity_total_time_s:.3f}s | "
            f"Intensity average time: {self.intensity_average_time_s:.3f}s"
        )
        if self.unmatched_images:
            lines.append(f"Unmatched images: {len(self.unmatched_images)}")
        if self.unmatched_pointclouds:
            lines.append(f"Unmatched point clouds: {len(self.unmatched_pointclouds)}")
        return "\n".join(lines)

    def pretty_print(self) -> None:
        print("")
        print(self)


def _stringify_argument(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        module = getattr(value, "__module__", None) or "unknown_module"
        qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", "callable")
        return f"{module}.{qualname}"
    return repr(value)


def _format_generate_arguments(arguments: dict[str, Any]) -> str:
    lines = ["\n\nSparse2Dense.generate arguments:"]
    for key, value in arguments.items():
        lines.append(f"- {key}: {_stringify_argument(value)}")
    return "\n".join(lines)


def _write_info_file(
    output_root: Path,
    arguments: dict[str, Any],
    summary: ProjectSummary,
) -> None:
    lines = [
        "Sparse2Dense run information",
        "",
        "Arguments:",
    ]
    for key, value in arguments.items():
        lines.append(f"- {key}: {_stringify_argument(value)}")

    lines.extend(
        [
            "",
            "Results:",
            f"- output_root: {summary.output_root}",
            f"- device_used: {summary.device_used}",
            f"- pairs_found: {summary.total_pairs_found}",
            f"- depth_maps_generated: {summary.depth_generated}",
            f"- intensity_maps_generated: {summary.intensity_generated}",
            f"- debug_images_generated: {summary.debug_generated}",
            f"- total_wall_time_s: {summary.total_wall_time_s:.6f}",
            f"- depth_total_time_s: {summary.depth_total_time_s:.6f}",
            f"- depth_average_time_s: {summary.depth_average_time_s:.6f}",
            f"- intensity_total_time_s: {summary.intensity_total_time_s:.6f}",
            f"- intensity_average_time_s: {summary.intensity_average_time_s:.6f}",
            f"- unmatched_images: {len(summary.unmatched_images)}",
            f"- unmatched_pointclouds: {len(summary.unmatched_pointclouds)}",
        ]
    )

    info_path = output_root / "info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _list_supported_files(path: Path, valid_extensions: set[str]) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions
    )


def _build_stem_map(files: list[Path]) -> dict[str, Path]:
    stem_map: dict[str, Path] = {}
    for file_path in files:
        if file_path.stem in stem_map:
            raise ValueError(
                f"Duplicate stem '{file_path.stem}' found for {stem_map[file_path.stem]} and {file_path}."
            )
        stem_map[file_path.stem] = file_path
    return stem_map


def _resolve_input_pairs(image_input: str | Path, pointcloud_input: str | Path) -> tuple[list[InputPair], list[str], list[str]]:
    """
    English:
    Resolve file/folder inputs into matched image-point-cloud pairs.

    Portuguese:
    Resolve entradas arquivo/pasta em pares casados de imagem e point cloud.
    """

    image_input = Path(image_input)
    pointcloud_input = Path(pointcloud_input)

    if not image_input.exists():
        raise FileNotFoundError(f"Image input not found: {image_input}")
    if not pointcloud_input.exists():
        raise FileNotFoundError(f"Point cloud input not found: {pointcloud_input}")

    if image_input.is_file() and pointcloud_input.is_file():
        return [InputPair(frame_id=image_input.stem, image_path=image_input, pointcloud_path=pointcloud_input)], [], []

    if image_input.is_file() and pointcloud_input.is_dir():
        candidates = _build_stem_map(_list_supported_files(pointcloud_input, POINTCLOUD_EXTENSIONS))
        if image_input.stem not in candidates:
            raise FileNotFoundError(
                f"Could not find a point cloud matching image stem '{image_input.stem}' inside {pointcloud_input}."
            )
        return [InputPair(frame_id=image_input.stem, image_path=image_input, pointcloud_path=candidates[image_input.stem])], [], []

    if image_input.is_dir() and pointcloud_input.is_file():
        candidates = _build_stem_map(_list_supported_files(image_input, IMAGE_EXTENSIONS))
        if pointcloud_input.stem not in candidates:
            raise FileNotFoundError(
                f"Could not find an image matching point cloud stem '{pointcloud_input.stem}' inside {image_input}."
            )
        return [InputPair(frame_id=pointcloud_input.stem, image_path=candidates[pointcloud_input.stem], pointcloud_path=pointcloud_input)], [], []

    if image_input.is_dir() and pointcloud_input.is_dir():
        image_files = _list_supported_files(image_input, IMAGE_EXTENSIONS)
        pointcloud_files = _list_supported_files(pointcloud_input, POINTCLOUD_EXTENSIONS)
        image_map = _build_stem_map(image_files)
        pointcloud_map = _build_stem_map(pointcloud_files)

        common_stems = sorted(set(image_map).intersection(pointcloud_map))
        if not common_stems:
            raise RuntimeError(
                f"No matching stems were found between {image_input} and {pointcloud_input}."
            )

        pairs = [
            InputPair(frame_id=stem, image_path=image_map[stem], pointcloud_path=pointcloud_map[stem])
            for stem in common_stems
        ]
        unmatched_images = sorted(set(image_map) - set(pointcloud_map))
        unmatched_pointclouds = sorted(set(pointcloud_map) - set(image_map))
        return pairs, unmatched_images, unmatched_pointclouds

    raise ValueError(
        "image_input and pointcloud_input must each be either a file or a folder."
    )


def _prepare_output_dirs(project_root: Path, depth: bool, intensity: bool, debug: bool) -> tuple[Path | None, Path | None, Path | None]:
    depth_dir = project_root / "depth_map" if depth else None
    intensity_dir = project_root / "intensity_map" if intensity else None
    debug_dir = project_root / "debug_images" if debug else None

    project_root.mkdir(parents=True, exist_ok=True)
    for folder in (depth_dir, intensity_dir, debug_dir):
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)

    return depth_dir, intensity_dir, debug_dir


def _compute_depth_and_intensity(
    prepared,
    *,
    device_used: str,
    depth: bool,
    intensity: bool,
    depth_mask_size: int,
    intensity_mask_size: int,
    ratio_threshold: float,
    jump_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """
    English:
    Compute depth and/or intensity, reusing a single run when mask sizes match.

    Portuguese:
    Calcula depth e/ou intensity, reutilizando uma unica passada quando as mascaras coincidem.
    """

    depth_map_m: np.ndarray | None = None
    intensity_map_raw: np.ndarray | None = None
    depth_time_s = 0.0
    intensity_time_s = 0.0
    densify_fn = densify_projection_torch if device_used == "cuda" else densify_projection

    def timed_densify(mask_size: int):
        if device_used == "cuda":
            torch_cuda_synchronize()
        start = time.perf_counter()
        dense_result = densify_fn(
            prepared,
            window_size=mask_size,
            ratio_threshold=ratio_threshold,
            jump_threshold=jump_threshold,
        )
        if device_used == "cuda":
            torch_cuda_synchronize()
        elapsed = time.perf_counter() - start
        return dense_result, elapsed

    if depth and intensity and depth_mask_size == intensity_mask_size:
        dense, elapsed = timed_densify(depth_mask_size)
        depth_map_m = dense.depth_map_m
        intensity_map_raw = dense.intensity_map_raw
        depth_time_s = elapsed
        intensity_time_s = elapsed
        return depth_map_m, intensity_map_raw, depth_time_s, intensity_time_s

    if depth:
        dense, depth_time_s = timed_densify(depth_mask_size)
        depth_map_m = dense.depth_map_m

    if intensity:
        dense, intensity_time_s = timed_densify(intensity_mask_size)
        intensity_map_raw = dense.intensity_map_raw

    return depth_map_m, intensity_map_raw, depth_time_s, intensity_time_s


def _resolve_device(device: str | None) -> str:
    if device is not None:
        requested_device = device.lower()
    else:
        requested_device = "cuda" if torch_cuda_available() else "cpu"

    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    if requested_device == "cuda" and not torch_cuda_available():
        raise RuntimeError("device='cuda' was requested, but PyTorch CUDA is not available.")
    return requested_device


def generate(
    *,
    image_input: str | Path,
    pointcloud_input: str | Path,
    calibration_yaml: str | Path,
    output_folder: str | Path,
    depth_mask_size: int = 13,
    intensity_mask_size: int = 13,
    depth: bool = True,
    intensity: bool = True,
    debug: bool = True,
    ratio_threshold: float = 0.15,
    jump_threshold: float = 0.01,
    device: str | None = None,
) -> ProjectSummary:
    """
    English:
    High-level Sparse2Dense API. It accepts single files or folders and creates
    a project folder with depth, intensity and debug outputs.

    Portuguese:
    API de alto nivel do Sparse2Dense. Aceita arquivos unicos ou pastas e cria
    uma pasta de projeto com saidas de depth, intensity e debug.
    """

    if not depth and not intensity and not debug:
        raise ValueError("At least one of depth, intensity or debug must be enabled.")
    if depth_mask_size < 1 or depth_mask_size % 2 == 0:
        raise ValueError("depth_mask_size must be a positive odd integer.")
    if intensity_mask_size < 1 or intensity_mask_size % 2 == 0:
        raise ValueError("intensity_mask_size must be a positive odd integer.")
    device_used = _resolve_device(device)

    start_arguments = {
        "image_input": image_input,
        "pointcloud_input": pointcloud_input,
        "calibration_yaml": calibration_yaml,
        "output_folder": output_folder,
        "depth_mask_size": depth_mask_size,
        "intensity_mask_size": intensity_mask_size,
        "depth": depth,
        "intensity": intensity,
        "debug": debug,
        "ratio_threshold": ratio_threshold,
        "jump_threshold": jump_threshold,
        "device": device if device is not None else device_used,
    }
    print(_format_generate_arguments(start_arguments))
    print("")

    calibration: CameraLidarCalibration = load_calibration(calibration_yaml)
    output_root = Path(output_folder)
    depth_dir, intensity_dir, debug_dir = _prepare_output_dirs(output_root, depth, intensity, debug)
    pairs, unmatched_images, unmatched_pointclouds = _resolve_input_pairs(image_input, pointcloud_input)

    total_wall_start = time.perf_counter()
    depth_total_time_s = 0.0
    intensity_total_time_s = 0.0
    depth_generated = 0
    intensity_generated = 0
    debug_generated = 0

    for pair in pairs:
        # Step 1 / Etapa 1:
        # Load the image and point cloud from disk.
        # Carrega a imagem e a point cloud do disco.
        rgb_image = load_image(pair.image_path)
        points_xyzi = load_pointcloud(pair.pointcloud_path)

        # Step 2 / Etapa 2:
        # Project the cloud once and build the reusable search structures.
        # Projeta a nuvem uma vez e monta as estruturas reutilizaveis de busca.
        prepared = prepare_projection(
            rgb_image=rgb_image,
            points_xyzi=points_xyzi,
            calibration=calibration,
        ) if device_used == "cpu" else prepare_projection_torch(
            rgb_image=rgb_image,
            points_xyzi=points_xyzi,
            calibration=calibration,
            device=device_used,
            keep_projected_points_cpu=debug,
        )

        # Step 3 / Etapa 3:
        # Run densification for depth and intensity. When mask sizes match,
        # the framework reuses a single pass.
        # Executa a densificacao para depth e intensity. Quando as mascaras
        # coincidem, o framework reutiliza uma unica passada.
        depth_map_m, intensity_map_raw, depth_time_s, intensity_time_s = _compute_depth_and_intensity(
            prepared,
            device_used=device_used,
            depth=depth,
            intensity=intensity,
            depth_mask_size=depth_mask_size,
            intensity_mask_size=intensity_mask_size,
            ratio_threshold=ratio_threshold,
            jump_threshold=jump_threshold,
        )
        depth_total_time_s += depth_time_s
        intensity_total_time_s += intensity_time_s

        # Step 4 / Etapa 4:
        # Persist outputs on disk using the project folder layout.
        # Persiste as saidas em disco usando o layout da pasta do projeto.
        if depth and depth_dir is not None and depth_map_m is not None:
            save_depth_png_mm(depth_map_m, depth_dir / f"{pair.frame_id}.png", invalid_value=0)
            depth_generated += 1

        if intensity and intensity_dir is not None and intensity_map_raw is not None:
            save_intensity_png(intensity_map_raw, intensity_dir / f"{pair.frame_id}.png", invalid_value=0)
            intensity_generated += 1

        if debug and debug_dir is not None:
            debug_image = make_debug_image(
                rgb_bgr=prepared.rgb_image,
                uv=prepared.projected_points_uv,
                depth_map_m=depth_map_m,
                intensity_map_raw=intensity_map_raw,
            )
            cv2.imwrite(str(debug_dir / f"{pair.frame_id}.png"), debug_image)
            debug_generated += 1

    total_wall_time_s = time.perf_counter() - total_wall_start
    summary = ProjectSummary(
        output_root=str(output_root),
        device_used=device_used,
        total_pairs_found=len(pairs),
        depth_generated=depth_generated,
        intensity_generated=intensity_generated,
        debug_generated=debug_generated,
        unmatched_images=unmatched_images,
        unmatched_pointclouds=unmatched_pointclouds,
        shared_compute_reused=bool(depth and intensity and depth_mask_size == intensity_mask_size),
        total_wall_time_s=total_wall_time_s,
        depth_total_time_s=depth_total_time_s,
        depth_average_time_s=depth_total_time_s / max(depth_generated, 1),
        intensity_total_time_s=intensity_total_time_s,
        intensity_average_time_s=intensity_total_time_s / max(intensity_generated, 1),
    )
    _write_info_file(output_root, start_arguments, summary)
    return summary


def run_project(**kwargs) -> ProjectSummary:
    """
    English:
    Backward-compatible alias. Prefer `generate(...)`.

    Portuguese:
    Alias para compatibilidade retroativa. Prefira `generate(...)`.
    """

    return generate(**kwargs)
