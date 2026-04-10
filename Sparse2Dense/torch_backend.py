from __future__ import annotations

"""
English:
Optional PyTorch + CUDA backend for Sparse2Dense.

Portuguese:
Backend opcional em PyTorch + CUDA para o Sparse2Dense.
"""

from dataclasses import dataclass

import numpy as np

from .calibration import CameraLidarCalibration
from .pipeline import DenseMaps

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


_CALIBRATION_TENSOR_CACHE: dict[tuple[str, bytes, bytes], tuple["torch.Tensor", "torch.Tensor"]] = {}
_SPATIAL_WEIGHTS_CACHE: dict[tuple[str, int], "torch.Tensor"] = {}


def torch_cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def ensure_torch_cuda_available() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not available. Install torch with CUDA support.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current PyTorch installation.")


def torch_cuda_synchronize() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass(frozen=True)
class TorchPreparedProjection:
    """
    English:
    GPU-friendly representation used by the PyTorch backend.

    Portuguese:
    Representacao amigavel para GPU usada pelo backend PyTorch.
    """

    rgb_image: np.ndarray
    projected_points_uv: np.ndarray
    sparse_depth_m: "torch.Tensor"
    sparse_intensity_raw: "torch.Tensor"
    device: str


def _get_calibration_tensors(
    calibration: CameraLidarCalibration,
    device: str,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    intrinsics = np.asarray(calibration.intrinsics_4x4, dtype=np.float32)
    lidar_to_camera = np.asarray(calibration.lidar_to_camera_4x4, dtype=np.float32)
    cache_key = (device, intrinsics.tobytes(), lidar_to_camera.tobytes())
    if cache_key not in _CALIBRATION_TENSOR_CACHE:
        _CALIBRATION_TENSOR_CACHE[cache_key] = (
            torch.as_tensor(lidar_to_camera, dtype=torch.float32, device=device),
            torch.as_tensor(intrinsics, dtype=torch.float32, device=device),
        )
    return _CALIBRATION_TENSOR_CACHE[cache_key]


def _get_spatial_weights(device: str, window_size: int) -> "torch.Tensor":
    cache_key = (device, window_size)
    if cache_key not in _SPATIAL_WEIGHTS_CACHE:
        radius = window_size // 2
        offset_y, offset_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=device, dtype=torch.float32),
            torch.arange(-radius, radius + 1, device=device, dtype=torch.float32),
            indexing="ij",
        )
        spatial_weights = 1.0 / (1.0 + torch.sqrt(offset_x.square() + offset_y.square()))
        _SPATIAL_WEIGHTS_CACHE[cache_key] = spatial_weights.reshape(1, -1)
    return _SPATIAL_WEIGHTS_CACHE[cache_key]


def _project_points_torch(
    points_xyzi: np.ndarray,
    calibration: CameraLidarCalibration,
    image_shape: tuple[int, int],
    device: str,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    height, width = image_shape
    points = torch.as_tensor(points_xyzi, dtype=torch.float32, device=device)
    ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device=device)
    points_h = torch.cat([points[:, :3], ones], dim=1)

    lidar_to_camera, intrinsics = _get_calibration_tensors(calibration, device)

    camera_points = points_h @ lidar_to_camera.T
    projected = camera_points @ intrinsics.T
    z_camera = projected[:, 2]
    uv = projected[:, :2] / torch.clamp(z_camera[:, None], min=1e-9)

    valid = (
        (z_camera > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )

    return uv[valid], z_camera[valid], points[valid, 3]


def _build_sparse_maps_torch(
    uv: "torch.Tensor",
    depth_m: "torch.Tensor",
    intensity_raw: "torch.Tensor",
    width: int,
    height: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    u_round = torch.round(uv[:, 0]).to(torch.int64)
    v_round = torch.round(uv[:, 1]).to(torch.int64)
    keep = (u_round >= 0) & (u_round < width) & (v_round >= 0) & (v_round < height)

    u_round = u_round[keep]
    v_round = v_round[keep]
    depth_m = depth_m[keep]
    intensity_raw = intensity_raw[keep]

    linear_idx = v_round * width + u_round

    if linear_idx.numel() == 0:
        sparse_depth = torch.full((height, width), float("nan"), dtype=torch.float32, device=uv.device)
        sparse_intensity = torch.full((height, width), float("nan"), dtype=torch.float32, device=uv.device)
        return sparse_depth, sparse_intensity

    depth_order = torch.argsort(depth_m, dim=0, descending=False, stable=True)
    linear_idx = linear_idx[depth_order]
    depth_m = depth_m[depth_order]
    intensity_raw = intensity_raw[depth_order]

    linear_order = torch.argsort(linear_idx, dim=0, descending=False, stable=True)
    linear_idx = linear_idx[linear_order]
    depth_m = depth_m[linear_order]
    intensity_raw = intensity_raw[linear_order]

    unique_idx, counts = torch.unique_consecutive(linear_idx, return_counts=True)
    first_positions = torch.cat(
        [
            torch.zeros((1,), dtype=torch.int64, device=uv.device),
            torch.cumsum(counts, dim=0)[:-1],
        ],
        dim=0,
    )

    sparse_depth_flat = torch.full(
        (height * width,),
        float("nan"),
        dtype=torch.float32,
        device=uv.device,
    )
    sparse_intensity_flat = torch.full(
        (height * width,),
        float("nan"),
        dtype=torch.float32,
        device=uv.device,
    )

    sparse_depth_flat[unique_idx] = depth_m[first_positions]
    sparse_intensity_flat[unique_idx] = intensity_raw[first_positions]

    return sparse_depth_flat.view(height, width), sparse_intensity_flat.view(height, width)


def prepare_projection_torch(
    rgb_image: np.ndarray,
    points_xyzi: np.ndarray,
    calibration: CameraLidarCalibration,
    *,
    device: str = "cuda",
    keep_projected_points_cpu: bool = True,
) -> TorchPreparedProjection:
    ensure_torch_cuda_available()
    with torch.inference_mode():
        height, width = rgb_image.shape[:2]
        uv, depth_m, intensity_raw = _project_points_torch(
            points_xyzi=points_xyzi,
            calibration=calibration,
            image_shape=(height, width),
            device=device,
        )
        sparse_depth_m, sparse_intensity_raw = _build_sparse_maps_torch(
            uv=uv,
            depth_m=depth_m,
            intensity_raw=intensity_raw,
            width=width,
            height=height,
        )

    projected_points_uv = (
        uv.detach().cpu().numpy().astype(np.float32)
        if keep_projected_points_cpu
        else np.empty((0, 2), dtype=np.float32)
    )
    return TorchPreparedProjection(
        rgb_image=rgb_image,
        projected_points_uv=projected_points_uv,
        sparse_depth_m=sparse_depth_m,
        sparse_intensity_raw=sparse_intensity_raw,
        device=device,
    )


def _compute_tile_rows(width: int, window_size: int) -> int:
    target_values = 6_000_000
    kernel_size = window_size * window_size
    tile_rows = max(4, target_values // max(width * kernel_size, 1))
    return int(max(4, min(64, tile_rows)))


def densify_projection_torch(
    prepared: TorchPreparedProjection,
    *,
    window_size: int = 13,
    ratio_threshold: float = 0.15,
    jump_threshold: float = 0.01,
    tile_rows: int | None = None,
) -> DenseMaps:
    """
    English:
    CUDA implementation of the modified bilateral densification.

    Portuguese:
    Implementacao CUDA da densificacao bilateral modificada.
    """

    ensure_torch_cuda_available()
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    with torch.inference_mode():
        sparse_depth = prepared.sparse_depth_m
        sparse_intensity = prepared.sparse_intensity_raw
        device = sparse_depth.device
        height, width = sparse_depth.shape
        radius = window_size // 2
        tile_rows = tile_rows or _compute_tile_rows(width, window_size)

        depth_input = torch.where(
            torch.isfinite(sparse_depth),
            sparse_depth,
            torch.full_like(sparse_depth, float("inf")),
        )
        intensity_input = torch.where(
            torch.isfinite(sparse_intensity),
            sparse_intensity,
            torch.zeros_like(sparse_intensity),
        )
        valid_input = torch.isfinite(sparse_depth).to(torch.float32)

        padded_depth = F.pad(depth_input.unsqueeze(0).unsqueeze(0), (radius, radius, radius, radius), value=float("inf"))
        padded_intensity = F.pad(intensity_input.unsqueeze(0).unsqueeze(0), (radius, radius, radius, radius), value=0.0)
        padded_valid = F.pad(valid_input.unsqueeze(0).unsqueeze(0), (radius, radius, radius, radius), value=0.0)

        spatial_weights = _get_spatial_weights(str(device), window_size)

        depth_out = torch.full((height * width,), float("nan"), dtype=torch.float32, device=device)
        intensity_out = torch.full((height * width,), float("nan"), dtype=torch.float32, device=device)

        kernel_elems = window_size * window_size
        inf_depth = torch.tensor(float("inf"), dtype=torch.float32, device=device)

        for row_start in range(0, height, tile_rows):
            row_end = min(row_start + tile_rows, height)

            depth_slice = padded_depth[:, :, row_start : row_end + 2 * radius, :]
            intensity_slice = padded_intensity[:, :, row_start : row_end + 2 * radius, :]
            valid_slice = padded_valid[:, :, row_start : row_end + 2 * radius, :]

            depth_patches = F.unfold(depth_slice, kernel_size=window_size).squeeze(0).transpose(0, 1)
            intensity_patches = F.unfold(intensity_slice, kernel_size=window_size).squeeze(0).transpose(0, 1)
            valid_patches = (F.unfold(valid_slice, kernel_size=window_size).squeeze(0).transpose(0, 1) > 0.5)

            sorted_depths, order = torch.sort(depth_patches, dim=1, descending=False)
            sorted_valid = torch.gather(valid_patches, 1, order)
            sorted_intensity = torch.gather(intensity_patches, 1, order)
            sorted_spatial = torch.gather(spatial_weights.expand(depth_patches.shape[0], -1), 1, order)

            valid_counts = sorted_valid.sum(dim=1)
            safe_depths = torch.where(sorted_valid, sorted_depths, torch.zeros_like(sorted_depths))
            near_depth = torch.where(
                valid_counts[:, None] > 0,
                safe_depths[:, :1],
                torch.zeros((safe_depths.shape[0], 1), dtype=safe_depths.dtype, device=device),
            )

            prev_depth = safe_depths[:, :-1]
            curr_depth = safe_depths[:, 1:]
            pair_valid = sorted_valid[:, :-1] & sorted_valid[:, 1:]
            denom = prev_depth + curr_depth
            relative_jump = torch.where(
                denom == 0.0,
                torch.abs(prev_depth - curr_depth),
                torch.abs((prev_depth - curr_depth) / denom),
            )
            cluster_breaks = pair_valid & (relative_jump > jump_threshold)

            cluster_ids = torch.cumsum(
                torch.cat(
                    [
                        torch.zeros((safe_depths.shape[0], 1), dtype=torch.int64, device=device),
                        cluster_breaks.to(torch.int64),
                    ],
                    dim=1,
                ),
                dim=1,
            ).clamp_max(kernel_elems - 1)

            counts = torch.zeros(
                (safe_depths.shape[0], kernel_elems),
                dtype=torch.float32,
                device=device,
            )
            counts.scatter_add_(1, cluster_ids, sorted_valid.to(torch.float32))

            cluster_start_mask = sorted_valid & torch.cat(
                [
                    torch.ones((safe_depths.shape[0], 1), dtype=torch.bool, device=device),
                    cluster_breaks,
                ],
                dim=1,
            )
            start_depth_values = torch.where(
                cluster_start_mask,
                safe_depths,
                torch.full_like(safe_depths, inf_depth),
            )

            depth_by_cluster = torch.full(
                (safe_depths.shape[0], kernel_elems),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
            depth_by_cluster.scatter_reduce_(
                1,
                cluster_ids,
                start_depth_values,
                reduce="amin",
                include_self=True,
            )

            chosen_cluster = torch.zeros((safe_depths.shape[0],), dtype=torch.int64, device=device)
            if kernel_elems > 1:
                other_counts = counts[:, 1:]
                max_other_count, _ = torch.max(other_counts, dim=1)
                has_other = max_other_count > 0.0
                other_depths = depth_by_cluster[:, 1:]
                candidate_depths = torch.where(
                    other_counts == max_other_count[:, None],
                    other_depths,
                    torch.full_like(other_depths, float("inf")),
                )
                other_cluster = torch.argmin(candidate_depths, dim=1) + 1
                other_cluster_size = counts.gather(1, other_cluster[:, None]).squeeze(1)
                closest_cluster_size = counts[:, 0]
                ratio = torch.where(
                    other_cluster_size > 0.0,
                    closest_cluster_size / other_cluster_size,
                    torch.full_like(other_cluster_size, 1e9),
                )
                chosen_cluster = torch.where(
                    has_other & (ratio <= ratio_threshold),
                    other_cluster,
                    chosen_cluster,
                )

            range_weight = 1.0 / (1.0 + torch.abs(safe_depths - near_depth))
            total_weights = sorted_spatial * range_weight * sorted_valid.to(torch.float32)
            chosen_mask = sorted_valid & (cluster_ids == chosen_cluster[:, None])
            chosen_weights = total_weights * chosen_mask.to(torch.float32)

            depth_weight_sum = chosen_weights.sum(dim=1)
            depth_numerator = (chosen_weights * safe_depths).sum(dim=1)

            finite_intensity = torch.isfinite(sorted_intensity)
            intensity_weights = total_weights * (chosen_mask & finite_intensity).to(torch.float32)
            intensity_weight_sum = intensity_weights.sum(dim=1)
            safe_intensity = torch.where(finite_intensity, sorted_intensity, torch.zeros_like(sorted_intensity))
            intensity_numerator = (intensity_weights * safe_intensity).sum(dim=1)

            tile_depth = torch.where(
                depth_weight_sum > 0.0,
                depth_numerator / depth_weight_sum,
                torch.full_like(depth_weight_sum, float("nan")),
            )
            tile_intensity = torch.where(
                intensity_weight_sum > 0.0,
                intensity_numerator / intensity_weight_sum,
                torch.full_like(intensity_weight_sum, float("nan")),
            )

            flat_start = row_start * width
            flat_end = row_end * width
            depth_out[flat_start:flat_end] = tile_depth
            intensity_out[flat_start:flat_end] = tile_intensity

        return DenseMaps(
            depth_map_m=depth_out.view(height, width).detach().cpu().numpy().astype(np.float32),
            intensity_map_raw=intensity_out.view(height, width).detach().cpu().numpy().astype(np.float32),
        )
