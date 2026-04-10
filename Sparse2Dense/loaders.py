from __future__ import annotations

"""
English:
Point cloud loaders used by Sparse2Dense.

Portuguese:
Loaders de nuvem de pontos usados pelo Sparse2Dense.
"""

from pathlib import Path
import re
import numpy as np


_AUTO_DELIMITER_PATTERN = re.compile(r"[,\t;| ]+")
_INTENSITY_ALIASES = ("intensity", "reflectivity", "i")


def _ensure_xyzi(points: np.ndarray, source: str | Path) -> np.ndarray:
    """
    English:
    Normalize input arrays to Nx4 = [x, y, z, intensity].

    Portuguese:
    Normaliza arrays de entrada para Nx4 = [x, y, z, intensity].
    """

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError(
            f"{source} must produce a 2D array with shape Nx3 or Nx4, got {points.shape}."
        )

    if points.shape[1] == 3:
        intensity = np.full((points.shape[0], 1), np.nan, dtype=np.float32)
        points = np.concatenate([points, intensity], axis=1)

    return points.astype(np.float32, copy=False)


def _extract_columns(
    matrix: np.ndarray,
    columns: dict[str, int] | None,
    source: str | Path,
) -> np.ndarray:
    """
    English:
    Extract x/y/z/intensity columns from a generic numeric matrix.

    Portuguese:
    Extrai colunas x/y/z/intensity de uma matriz numerica generica.
    """

    if columns is None:
        columns = {"x": 0, "y": 1, "z": 2, "intensity": 3 if matrix.shape[1] >= 4 else -1}

    required = ("x", "y", "z")
    for key in required:
        if key not in columns:
            raise ValueError(f"{source} columns mapping must define '{key}'.")

    max_index = max(index for index in columns.values() if index >= 0)
    if max_index >= matrix.shape[1]:
        raise ValueError(
            f"{source} columns mapping requests column {max_index}, but the input has "
            f"only {matrix.shape[1]} columns."
        )

    xyz = matrix[:, [columns["x"], columns["y"], columns["z"]]].astype(np.float32)
    intensity_index = columns.get("intensity", -1)
    if intensity_index is None or intensity_index < 0:
        return _ensure_xyzi(xyz, source)

    intensity = matrix[:, intensity_index : intensity_index + 1].astype(np.float32)
    return _ensure_xyzi(np.concatenate([xyz, intensity], axis=1), source)


def load_bin_pointcloud(
    pointcloud_path: str | Path,
    *,
    dtype: np.dtype = np.float32,
    num_fields: int = 4,
    columns: dict[str, int] | None = None,
) -> np.ndarray:
    """
    English:
    Load a `.bin` point cloud exported as a flat numeric array.
    This is suitable for common KITTI/Velodyne/Ouster exports stored as
    float matrices. Raw packet captures are not decoded here.

    Portuguese:
    Carrega uma point cloud `.bin` exportada como um array numerico plano.
    Isso atende exportacoes comuns de KITTI/Velodyne/Ouster salvas como
    matrizes de float. Capturas cruas de pacotes nao sao decodificadas aqui.
    """

    pointcloud_path = Path(pointcloud_path)
    raw = np.fromfile(pointcloud_path, dtype=dtype)
    if num_fields <= 0:
        raise ValueError("num_fields must be a positive integer.")
    if raw.size % num_fields != 0:
        raise ValueError(
            f"{pointcloud_path} cannot be reshaped into Nx{num_fields}. "
            f"Total values found: {raw.size}."
        )

    matrix = raw.reshape(-1, num_fields)
    return _extract_columns(matrix, columns, pointcloud_path)


def _read_ply_header(pointcloud_path: Path) -> tuple[str, int, list[str], list[str], int]:
    with pointcloud_path.open("rb") as fh:
        first_line = fh.readline().decode("ascii", errors="strict").strip()
        if first_line != "ply":
            raise ValueError(f"{pointcloud_path} is not a PLY file.")

        fmt = ""
        vertex_count = -1
        property_names: list[str] = []
        property_types: list[str] = []
        in_vertex = False

        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{pointcloud_path} ended before end_header.")
            decoded = line.decode("ascii", errors="strict").strip()
            if decoded == "end_header":
                data_offset = fh.tell()
                break
            if not decoded:
                continue

            parts = decoded.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                property_types.append(parts[1])
                property_names.append(parts[2])

    if vertex_count < 0:
        raise ValueError(f"{pointcloud_path} does not declare a vertex count.")

    return fmt, vertex_count, property_names, property_types, data_offset


def load_ply_pointcloud(pointcloud_path: str | Path) -> np.ndarray:
    """
    English:
    Load an ASCII or binary little-endian PLY file containing x/y/z and,
    optionally, intensity-like fields.

    Portuguese:
    Carrega um arquivo PLY ASCII ou binario little-endian contendo x/y/z e,
    opcionalmente, campos equivalentes a intensidade.
    """

    pointcloud_path = Path(pointcloud_path)
    fmt, vertex_count, property_names, property_types, data_offset = _read_ply_header(pointcloud_path)

    intensity_name = next((name for name in _INTENSITY_ALIASES if name in property_names), None)
    required_names = ("x", "y", "z")
    for name in required_names:
        if name not in property_names:
            raise ValueError(f"{pointcloud_path} is missing required property '{name}'.")

    if fmt == "ascii":
        with pointcloud_path.open("rb") as fh:
            fh.seek(data_offset)
            matrix = np.loadtxt(fh, dtype=np.float32, max_rows=vertex_count)
        if matrix.ndim == 1:
            matrix = matrix[np.newaxis, :]
        name_to_index = {name: idx for idx, name in enumerate(property_names)}
        columns = {"x": name_to_index["x"], "y": name_to_index["y"], "z": name_to_index["z"]}
        columns["intensity"] = -1 if intensity_name is None else name_to_index[intensity_name]
        return _extract_columns(matrix, columns, pointcloud_path)

    if fmt != "binary_little_endian":
        raise ValueError(
            f"{pointcloud_path} uses unsupported PLY format {fmt!r}. "
            f"Only ascii and binary_little_endian are supported."
        )

    numpy_type_map = {
        "char": "i1",
        "uchar": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "float": "<f4",
        "double": "<f8",
    }

    dtype_fields = []
    for type_name, field_name in zip(property_types, property_names):
        if type_name not in numpy_type_map:
            raise ValueError(f"Unsupported PLY property type {type_name!r} in {pointcloud_path}.")
        dtype_fields.append((field_name, numpy_type_map[type_name]))

    with pointcloud_path.open("rb") as fh:
        fh.seek(data_offset)
        data = np.fromfile(fh, dtype=np.dtype(dtype_fields), count=vertex_count)

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    if intensity_name is None:
        return _ensure_xyzi(xyz, pointcloud_path)

    intensity = np.asarray(data[intensity_name], dtype=np.float32).reshape(-1, 1)
    return _ensure_xyzi(np.concatenate([xyz, intensity], axis=1), pointcloud_path)


def _build_pcd_dtype(fields: list[str], sizes: list[int], types: list[str], counts: list[int]) -> np.dtype:
    dtype_fields: list[tuple[str, str]] = []
    pcd_type_map = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }

    for name, size, type_name, count in zip(fields, sizes, types, counts):
        key = (type_name.upper(), int(size))
        if key not in pcd_type_map:
            raise ValueError(f"Unsupported PCD field type combination: {key}.")
        if int(count) == 1:
            dtype_fields.append((name, pcd_type_map[key]))
            continue
        for idx in range(int(count)):
            dtype_fields.append((f"{name}_{idx}", pcd_type_map[key]))

    return np.dtype(dtype_fields)


def load_pcd_pointcloud(pointcloud_path: str | Path) -> np.ndarray:
    """
    English:
    Load a `.pcd` file with x/y/z and optional intensity.
    Supports `DATA ascii` and `DATA binary`.

    Portuguese:
    Carrega um arquivo `.pcd` com x/y/z e intensidade opcional.
    Suporta `DATA ascii` e `DATA binary`.
    """

    pointcloud_path = Path(pointcloud_path)
    header_lines: list[str] = []
    with pointcloud_path.open("rb") as fh:
        while True:
            raw_line = fh.readline()
            if not raw_line:
                raise ValueError(f"{pointcloud_path} ended before the DATA section.")
            decoded = raw_line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded.upper().startswith("DATA "):
                data_offset = fh.tell()
                break

    header: dict[str, str] = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        key = parts[0].upper()
        value = "" if len(parts) == 1 else parts[1]
        header[key] = value

    fields = header.get("FIELDS", "").split()
    sizes = [int(value) for value in header.get("SIZE", "").split()]
    types = header.get("TYPE", "").split()
    counts = [int(value) for value in header.get("COUNT", " ".join(["1"] * len(fields))).split()]
    points_count = int(header.get("POINTS", header.get("WIDTH", "0")))
    data_mode = header.get("DATA", "").lower()

    if not fields or not sizes or not types:
        raise ValueError(f"{pointcloud_path} has an incomplete PCD header.")
    if data_mode == "binary_compressed":
        raise ValueError(
            f"{pointcloud_path} uses DATA binary_compressed, which is not supported yet."
        )
    if points_count <= 0:
        raise ValueError(f"{pointcloud_path} does not define a positive POINTS count.")

    intensity_name = next((name for name in _INTENSITY_ALIASES if name in fields), None)
    if "x" not in fields or "y" not in fields or "z" not in fields:
        raise ValueError(f"{pointcloud_path} is missing one of the required fields x/y/z.")

    if data_mode == "ascii":
        with pointcloud_path.open("rb") as fh:
            fh.seek(data_offset)
            matrix = np.loadtxt(fh, dtype=np.float32, max_rows=points_count)
        if matrix.ndim == 1:
            matrix = matrix[np.newaxis, :]
        name_to_index = {name: idx for idx, name in enumerate(fields)}
        columns = {"x": name_to_index["x"], "y": name_to_index["y"], "z": name_to_index["z"]}
        columns["intensity"] = -1 if intensity_name is None else name_to_index[intensity_name]
        return _extract_columns(matrix, columns, pointcloud_path)

    if data_mode != "binary":
        raise ValueError(
            f"{pointcloud_path} uses unsupported PCD DATA mode {data_mode!r}. "
            f"Only ascii and binary are supported."
        )

    dtype = _build_pcd_dtype(fields, sizes, types, counts)
    with pointcloud_path.open("rb") as fh:
        fh.seek(data_offset)
        data = np.fromfile(fh, dtype=dtype, count=points_count)

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    if intensity_name is None:
        return _ensure_xyzi(xyz, pointcloud_path)

    intensity = np.asarray(data[intensity_name], dtype=np.float32).reshape(-1, 1)
    return _ensure_xyzi(np.concatenate([xyz, intensity], axis=1), pointcloud_path)


def _tokenize_table_line(line: str, delimiter: str | None) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []

    if delimiter is None or delimiter == "auto":
        return [token for token in _AUTO_DELIMITER_PATTERN.split(stripped) if token]

    return [token.strip() for token in stripped.split(delimiter) if token.strip()]


def load_table_pointcloud(
    pointcloud_path: str | Path,
) -> np.ndarray:
    """
    English:
    Load `.txt` or `.csv` point clouds using automatic delimiter detection.
    The parser accepts common separators such as spaces, tabs, commas,
    semicolons and pipes.

    Portuguese:
    Carrega point clouds `.txt` ou `.csv` usando deteccao automatica de delimitador.
    O parser aceita separadores comuns como espaco, tab, virgula,
    ponto-e-virgula e pipe.
    """

    pointcloud_path = Path(pointcloud_path)
    numeric_rows: list[list[float]] = []
    expected_width: int | None = None

    with pointcloud_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = _tokenize_table_line(line, "auto")
            if len(tokens) < 3:
                continue

            try:
                numeric_row = [float(token) for token in tokens]
            except ValueError:
                # English: Skip header-like lines.
                # Portuguese: Ignora linhas parecidas com cabecalho.
                continue

            if expected_width is None:
                expected_width = len(numeric_row)
            if len(numeric_row) != expected_width:
                continue

            numeric_rows.append(numeric_row)

    if not numeric_rows:
        raise ValueError(f"{pointcloud_path} does not contain numeric point rows.")

    matrix = np.asarray(numeric_rows, dtype=np.float32)
    return _extract_columns(matrix, None, pointcloud_path)


def load_pointcloud(
    pointcloud_path: str | Path,
) -> np.ndarray:
    """
    English:
    Dispatch point cloud loading based on file extension.

    Portuguese:
    Despacha o carregamento da point cloud com base na extensao.
    """

    pointcloud_path = Path(pointcloud_path)

    suffix = pointcloud_path.suffix.lower()
    if suffix == ".bin":
        return load_bin_pointcloud(pointcloud_path)
    if suffix == ".ply":
        return load_ply_pointcloud(pointcloud_path)
    if suffix == ".pcd":
        return load_pcd_pointcloud(pointcloud_path)
    if suffix in {".txt", ".csv"}:
        return load_table_pointcloud(pointcloud_path)

    raise ValueError(
        f"Unsupported point cloud extension {pointcloud_path.suffix!r}. "
        f"Supported formats: .bin, .ply, .pcd, .txt, .csv. "
        f"Convert the file to .txt or .csv with columns x y z [intensity] and run again."
    )
