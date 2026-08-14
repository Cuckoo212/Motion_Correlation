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
        empty_colors = (
            None if rgb_bgr is None else np.empty((0, 3), dtype=np.uint8)
        )
        return np.empty((0, 3), np.float32), empty_colors

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
    """Save an ASCII PLY compatible with cable_camera_frame.ply examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None and len(colors) == len(points)
    color_header = (
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        if has_color
        else ""
    )
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"{color_header}"
        "end_header\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        if len(points):
            if has_color:
                vertices = np.column_stack(
                    (points.astype(np.float64), colors.astype(np.uint8))
                )
                np.savetxt(
                    stream,
                    vertices,
                    fmt=("%.6f", "%.6f", "%.6f", "%d", "%d", "%d"),
                )
            else:
                np.savetxt(stream, points, fmt="%.6f")
    temporary.replace(path)
