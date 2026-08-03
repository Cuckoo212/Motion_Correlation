"""RGB-D projection and point-cloud persistence utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def depth_mask_to_points(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    rgb_bgr: np.ndarray | None = None,
    *,
    min_depth_mm: int = 150,
    max_depth_mm: int = 2000,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project masked, aligned depth pixels into the color optical frame."""
    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a single-channel image")
    if mask.shape != depth_mm.shape:
        raise ValueError("mask and depth must have identical dimensions")
    if rgb_bgr is not None and rgb_bgr.shape[:2] != depth_mm.shape:
        raise ValueError("RGB and depth must have identical dimensions")

    valid = (
        (mask > 0)
        & np.isfinite(depth_mm)
        & (depth_mm >= min_depth_mm)
        & (depth_mm <= max_depth_mm)
    )
    v, u = np.nonzero(valid)
    if not len(u):
        return np.empty((0, 3), np.float32), None

    z = depth_mm[v, u].astype(np.float32) * 0.001
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    colors = None if rgb_bgr is None else rgb_bgr[v, u, ::-1].copy()
    return points, colors


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep one representative point per voxel without requiring Open3D."""
    if voxel_size_m <= 0 or len(points) < 2:
        return points, colors
    voxels = np.floor(points / voxel_size_m).astype(np.int64)
    _, indices = np.unique(voxels, axis=0, return_index=True)
    indices.sort()
    return points[indices], None if colors is None else colors[indices]


def save_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Save a compact binary little-endian PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None and len(colors) == len(points)
    dtype_fields: list[tuple[str, str]] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
    ]
    if has_color:
        dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(points), dtype=np.dtype(dtype_fields))
    for column, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, column]
    if has_color:
        for column, name in enumerate(("red", "green", "blue")):
            vertices[name] = colors[:, column]

    color_header = (
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        if has_color
        else ""
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{color_header}"
        "end_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)
