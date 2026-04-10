from __future__ import annotations

"""
English:
Utilities to read camera/LiDAR calibration files used by Sparse2Dense.

Portuguese:
Utilitarios para ler arquivos de calibracao camera/LiDAR usados pelo Sparse2Dense.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraLidarCalibration:
    """
    English:
    Minimal calibration model used by the framework.

    Portuguese:
    Modelo minimo de calibracao usado pelo framework.
    """

    intrinsics_4x4: np.ndarray
    lidar_to_camera_4x4: np.ndarray


def _ensure_matrix_shape(matrix: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}.")
    return matrix


def load_calibration(calibration_yaml: str | Path) -> CameraLidarCalibration:
    """
    English:
    Load a YAML file with the two matrices required by the framework:
    `intrinsics_4x4` and `lidar_to_camera_4x4`.

    Portuguese:
    Carrega um arquivo YAML com as duas matrizes exigidas pelo framework:
    `intrinsics_4x4` e `lidar_to_camera_4x4`.
    """

    calibration_yaml = Path(calibration_yaml)
    with calibration_yaml.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"{calibration_yaml} does not contain a valid YAML mapping.")

    intrinsics_4x4 = _ensure_matrix_shape(
        np.asarray(payload["intrinsics_4x4"], dtype=np.float64),
        (4, 4),
        "intrinsics_4x4",
    )
    lidar_to_camera_4x4 = _ensure_matrix_shape(
        np.asarray(payload["lidar_to_camera_4x4"], dtype=np.float64),
        (4, 4),
        "lidar_to_camera_4x4",
    )

    return CameraLidarCalibration(
        intrinsics_4x4=intrinsics_4x4,
        lidar_to_camera_4x4=lidar_to_camera_4x4,
    )
