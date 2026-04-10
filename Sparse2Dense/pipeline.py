from __future__ import annotations

"""
English:
Core projection and densification pipeline for Sparse2Dense.

Portuguese:
Pipeline principal de projecao e densificacao do Sparse2Dense.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .calibration import CameraLidarCalibration

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


MAX_LOCAL_POINTS = 4096


@dataclass(frozen=True)
class PreparedProjection:
    """
    English:
    Stores the projected point cloud and the structures reused by the filter.

    Portuguese:
    Armazena a point cloud projetada e as estruturas reutilizadas pelo filtro.
    """

    rgb_image: np.ndarray
    projected_points_uv: np.ndarray
    sparse_depth_m: np.ndarray
    sparse_intensity_raw: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    depths: np.ndarray
    intensities: np.ndarray
    bin_start: np.ndarray
    bin_count: np.ndarray


@dataclass(frozen=True)
class DenseMaps:
    depth_map_m: np.ndarray
    intensity_map_raw: np.ndarray


def load_image(image_path: str | Path) -> np.ndarray:
    """
    English:
    Read an RGB image as BGR, which is OpenCV's native layout.

    Portuguese:
    Le uma imagem RGB como BGR, que e o layout nativo do OpenCV.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def project_pointcloud_to_image(
    points_xyzi: np.ndarray,
    calibration: CameraLidarCalibration,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    English:
    Project LiDAR points to the image plane and keep only valid pixels.

    Portuguese:
    Projeta pontos LiDAR no plano da imagem e mantem apenas pixels validos.
    """

    height, width = image_shape
    num_points = points_xyzi.shape[0]
    points_h = np.concatenate(
        [points_xyzi[:, :3], np.ones((num_points, 1), dtype=np.float32)],
        axis=1,
    ).astype(np.float64)

    camera_points = points_h @ calibration.lidar_to_camera_4x4.T
    projected = camera_points @ calibration.intrinsics_4x4.T
    z_camera = projected[:, 2]
    uv = projected[:, :2] / np.maximum(z_camera[:, None], 1e-9)

    valid = (
        (z_camera > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )

    return (
        uv[valid].astype(np.float32),
        z_camera[valid].astype(np.float32),
        points_xyzi[valid, 3].astype(np.float32),
    )


def _build_sparse_maps(
    uv: np.ndarray,
    depth_m: np.ndarray,
    intensity_raw: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    sparse_depth = np.full((height, width), np.nan, dtype=np.float32)
    sparse_intensity = np.full((height, width), np.nan, dtype=np.float32)

    u_round = np.rint(uv[:, 0]).astype(np.int32)
    v_round = np.rint(uv[:, 1]).astype(np.int32)
    keep = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)

    for uu, vv, zz, ii in zip(u_round[keep], v_round[keep], depth_m[keep], intensity_raw[keep]):
        old_depth = sparse_depth[vv, uu]
        if np.isnan(old_depth) or zz < old_depth:
            sparse_depth[vv, uu] = zz
            sparse_intensity[vv, uu] = ii

    return sparse_depth, sparse_intensity


def _build_bins(
    uv: np.ndarray,
    depth_m: np.ndarray,
    intensity_raw: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_round = np.rint(uv[:, 0]).astype(np.int32)
    v_round = np.rint(uv[:, 1]).astype(np.int32)
    keep = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)

    u_round = u_round[keep]
    v_round = v_round[keep]
    uv = uv[keep]
    depth_m = depth_m[keep]
    intensity_raw = intensity_raw[keep]

    bin_index = v_round.astype(np.int64) * width + u_round.astype(np.int64)
    order = np.argsort(bin_index, kind="mergesort")

    sorted_bins = bin_index[order].astype(np.int32)
    sorted_u = uv[order, 0].astype(np.float32)
    sorted_v = uv[order, 1].astype(np.float32)
    sorted_depth = depth_m[order].astype(np.float32)
    sorted_intensity = intensity_raw[order].astype(np.float32)

    bin_start = np.full(width * height, -1, dtype=np.int32)
    bin_count = np.zeros(width * height, dtype=np.int32)

    if sorted_bins.size > 0:
        last_bin = int(sorted_bins[0])
        bin_start[last_bin] = 0
        bin_count[last_bin] = 1
        for idx in range(1, sorted_bins.size):
            current_bin = int(sorted_bins[idx])
            if current_bin != last_bin:
                bin_start[current_bin] = idx
                last_bin = current_bin
            bin_count[current_bin] += 1

    return sorted_u, sorted_v, sorted_depth, sorted_intensity, bin_start, bin_count


def prepare_projection(
    rgb_image: np.ndarray,
    points_xyzi: np.ndarray,
    calibration: CameraLidarCalibration,
) -> PreparedProjection:
    """
    English:
    Project the cloud once and pre-build sparse maps plus fast pixel bins.

    Portuguese:
    Projeta a nuvem uma unica vez e preconstroi mapas esparsos e bins rapidos por pixel.
    """

    height, width = rgb_image.shape[:2]
    uv, depth_m, intensity_raw = project_pointcloud_to_image(
        points_xyzi=points_xyzi,
        calibration=calibration,
        image_shape=(height, width),
    )
    sparse_depth_m, sparse_intensity_raw = _build_sparse_maps(
        uv=uv,
        depth_m=depth_m,
        intensity_raw=intensity_raw,
        width=width,
        height=height,
    )
    xs, ys, depths, intensities, bin_start, bin_count = _build_bins(
        uv=uv,
        depth_m=depth_m,
        intensity_raw=intensity_raw,
        width=width,
        height=height,
    )

    return PreparedProjection(
        rgb_image=rgb_image,
        projected_points_uv=uv,
        sparse_depth_m=sparse_depth_m,
        sparse_intensity_raw=sparse_intensity_raw,
        xs=xs,
        ys=ys,
        depths=depths,
        intensities=intensities,
        bin_start=bin_start,
        bin_count=bin_count,
    )


@njit(cache=True)
def _dense_modified_bilateral(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    intensities: np.ndarray,
    bin_start: np.ndarray,
    bin_count: np.ndarray,
    width: int,
    height: int,
    window_size: int,
    ratio_threshold: float,
    jump_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = window_size // 2
    out_depth = np.empty((height, width), dtype=np.float32)
    out_intensity = np.empty((height, width), dtype=np.float32)
    out_depth[:] = np.nan
    out_intensity[:] = np.nan

    local_x = np.empty(MAX_LOCAL_POINTS, dtype=np.float32)
    local_y = np.empty(MAX_LOCAL_POINTS, dtype=np.float32)
    local_depth = np.empty(MAX_LOCAL_POINTS, dtype=np.float32)
    local_intensity = np.empty(MAX_LOCAL_POINTS, dtype=np.float32)
    cluster_starts = np.empty(MAX_LOCAL_POINTS, dtype=np.int32)
    cluster_ends = np.empty(MAX_LOCAL_POINTS, dtype=np.int32)

    for v in range(height):
        y0 = 0 if v < radius else v - radius
        y1 = height - 1 if v + radius >= height else v + radius

        for u in range(width):
            x0 = 0 if u < radius else u - radius
            x1 = width - 1 if u + radius >= width else u + radius

            local_count = 0

            for yy in range(y0, y1 + 1):
                row_offset = yy * width
                for xx in range(x0, x1 + 1):
                    flat_idx = row_offset + xx
                    count = bin_count[flat_idx]
                    if count == 0:
                        continue

                    start = bin_start[flat_idx]
                    for pos in range(start, start + count):
                        if local_count >= MAX_LOCAL_POINTS:
                            break
                        local_x[local_count] = xs[pos]
                        local_y[local_count] = ys[pos]
                        local_depth[local_count] = depths[pos]
                        local_intensity[local_count] = intensities[pos]
                        local_count += 1

            if local_count == 0:
                continue

            order = np.argsort(local_depth[:local_count])
            cluster_count = 1
            cluster_starts[0] = 0
            last_depth = local_depth[order[0]]
            near_depth = last_depth

            for idx in range(local_count):
                current_depth = local_depth[order[idx]]
                denom = last_depth + current_depth
                relative_jump = abs(last_depth - current_depth) if denom == 0.0 else abs((last_depth - current_depth) / denom)
                if relative_jump > jump_threshold:
                    cluster_ends[cluster_count - 1] = idx
                    cluster_starts[cluster_count] = idx
                    cluster_count += 1
                last_depth = current_depth
            cluster_ends[cluster_count - 1] = local_count

            chosen_cluster = 0
            if cluster_count > 1:
                other_cluster = 1
                best_other_size = cluster_ends[1] - cluster_starts[1]
                best_other_depth = local_depth[order[cluster_starts[1]]]

                for cluster_idx in range(2, cluster_count):
                    cluster_size = cluster_ends[cluster_idx] - cluster_starts[cluster_idx]
                    cluster_depth = local_depth[order[cluster_starts[cluster_idx]]]
                    if cluster_size > best_other_size or (
                        cluster_size == best_other_size and cluster_depth < best_other_depth
                    ):
                        other_cluster = cluster_idx
                        best_other_size = cluster_size
                        best_other_depth = cluster_depth

                closest_cluster_size = cluster_ends[0] - cluster_starts[0]
                denominator = cluster_ends[other_cluster] - cluster_starts[other_cluster]
                ratio = 1e9 if denominator <= 0 else closest_cluster_size / denominator
                chosen_cluster = 0 if ratio > ratio_threshold else other_cluster

            start = cluster_starts[chosen_cluster]
            end = cluster_ends[chosen_cluster]
            weight_sum = 0.0
            depth_acc = 0.0
            intensity_acc = 0.0

            for pos in range(start, end):
                idx = order[pos]
                dx = float(u) - float(local_x[idx])
                dy = float(v) - float(local_y[idx])
                spatial_weight = 1.0 / (1.0 + np.sqrt(dx * dx + dy * dy))
                range_weight = 1.0 / (1.0 + abs(float(near_depth) - float(local_depth[idx])))
                weight = spatial_weight * range_weight

                weight_sum += weight
                depth_acc += weight * float(local_depth[idx])
                intensity_acc += weight * float(local_intensity[idx])

            if weight_sum > 0.0:
                out_depth[v, u] = np.float32(depth_acc / weight_sum)
                out_intensity[v, u] = np.float32(intensity_acc / weight_sum)

    return out_depth, out_intensity


def densify_projection(
    prepared: PreparedProjection,
    *,
    window_size: int = 13,
    ratio_threshold: float = 0.15,
    jump_threshold: float = 0.01,
) -> DenseMaps:
    """
    English:
    Run the modified bilateral filter on a prepared projection.

    Portuguese:
    Executa o filtro bilateral modificado sobre uma projecao ja preparada.
    """

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    height, width = prepared.rgb_image.shape[:2]
    depth_map_m, intensity_map_raw = _dense_modified_bilateral(
        xs=prepared.xs,
        ys=prepared.ys,
        depths=prepared.depths,
        intensities=prepared.intensities,
        bin_start=prepared.bin_start,
        bin_count=prepared.bin_count,
        width=width,
        height=height,
        window_size=window_size,
        ratio_threshold=ratio_threshold,
        jump_threshold=jump_threshold,
    )
    return DenseMaps(depth_map_m=depth_map_m, intensity_map_raw=intensity_map_raw)


def save_depth_png_mm(depth_map_m: np.ndarray, output_path: str | Path, invalid_value: int = 0) -> np.ndarray:
    """
    English:
    Save depth in millimetres as uint16 PNG.

    Portuguese:
    Salva profundidade em milimetros como PNG uint16.
    """

    output_path = Path(output_path)
    depth_mm = np.where(np.isnan(depth_map_m), float(invalid_value), depth_map_m * 1000.0)
    depth_uint16 = np.clip(np.rint(depth_mm), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(depth_uint16).save(output_path)
    return depth_uint16


def save_intensity_png(intensity_map_raw: np.ndarray, output_path: str | Path, invalid_value: int = 0) -> np.ndarray:
    """
    English:
    Save intensity as a visualization-friendly uint16 PNG.
    The function normalizes valid values to the available 16-bit range.

    Portuguese:
    Salva intensidade como PNG uint16 amigavel para visualizacao.
    A funcao normaliza valores validos para o intervalo disponivel em 16 bits.
    """

    output_path = Path(output_path)
    valid = np.isfinite(intensity_map_raw)
    output = np.full(intensity_map_raw.shape, np.uint16(invalid_value), dtype=np.uint16)
    if not np.any(valid):
        Image.fromarray(output).save(output_path)
        return output

    valid_values = intensity_map_raw[valid].astype(np.float32)
    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))

    if max_value <= min_value:
        output[valid] = np.uint16(np.iinfo(np.uint16).max)
    else:
        normalized = (valid_values - min_value) / (max_value - min_value)
        output[valid] = np.clip(
            np.rint(normalized * np.iinfo(np.uint16).max),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

    Image.fromarray(output).save(output_path)
    return output


def colorize_depth(depth_map_m: np.ndarray, max_depth_m: float = 80.0) -> np.ndarray:
    valid = np.isfinite(depth_map_m)
    normalized = np.zeros(depth_map_m.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_map_m, 0.0, max_depth_m)
        normalized[valid] = np.rint(clipped[valid] / max_depth_m * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    color[~valid] = 0
    return color


def colorize_intensity(intensity_map_raw: np.ndarray) -> np.ndarray:
    valid = np.isfinite(intensity_map_raw)
    normalized = np.zeros(intensity_map_raw.shape, dtype=np.uint8)
    if np.any(valid):
        values = intensity_map_raw[valid].astype(np.float32)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value <= min_value:
            normalized[valid] = 255
        else:
            scaled = (values - min_value) / (max_value - min_value)
            normalized[valid] = np.rint(scaled * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    color[~valid] = 0
    return color


def overlay_projected_points(rgb_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    overlay = rgb_bgr.copy()
    if uv.size == 0:
        return overlay

    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    keep = (u >= 0) & (u < overlay.shape[1]) & (v >= 0) & (v < overlay.shape[0])
    overlay[v[keep], u[keep]] = (0, 0, 255)
    return overlay


def annotate_panel(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    text_x = 14
    text_y = 28
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    top_left = (max(text_x - 8, 0), max(text_y - text_height - 8, 0))
    bottom_right = (
        min(text_x + text_width + 8, out.shape[1] - 1),
        min(text_y + baseline + 8, out.shape[0] - 1),
    )
    cv2.rectangle(out, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def make_debug_image(
    rgb_bgr: np.ndarray,
    uv: np.ndarray,
    depth_map_m: np.ndarray | None,
    intensity_map_raw: np.ndarray | None,
) -> np.ndarray:
    """
    English:
    Choose the debug layout automatically from the image aspect ratio.
    Very wide images are stacked vertically, while more compact images use a 2x2 grid.

    Portuguese:
    Escolhe o layout do debug automaticamente pela proporcao da imagem.
    Imagens muito largas ficam empilhadas verticalmente, enquanto imagens mais compactas usam uma grade 2x2.
    """

    panels = [
        annotate_panel(rgb_bgr, "Image"),
        annotate_panel(overlay_projected_points(rgb_bgr, uv), "Point cloud over image"),
    ]

    if depth_map_m is None:
        empty = np.zeros_like(rgb_bgr)
        panels.append(annotate_panel(empty, "Depth disabled"))
    else:
        panels.append(annotate_panel(colorize_depth(depth_map_m), "Depth map"))

    if intensity_map_raw is None:
        empty = np.zeros_like(rgb_bgr)
        panels.append(annotate_panel(empty, "Intensity disabled"))
    else:
        panels.append(annotate_panel(colorize_intensity(intensity_map_raw), "Intensity map"))

    height, width = rgb_bgr.shape[:2]
    aspect_ratio = float(width) / float(max(height, 1))
    if aspect_ratio < 1.5:
        top_row = np.concatenate(panels[:2], axis=1)
        bottom_row = np.concatenate(panels[2:], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)

    return np.concatenate(panels, axis=0)
