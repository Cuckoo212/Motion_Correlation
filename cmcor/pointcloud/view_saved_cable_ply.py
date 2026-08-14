#!/usr/bin/env python3
"""Interactively inspect one saved camera-frame cable PLY with Matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load x/y/z and optional RGB fields from this project's ASCII PLY files."""
    with path.open("r", encoding="ascii") as stream:
        properties: list[str] = []
        vertex_count: int | None = None
        in_vertex = False
        for raw_line in stream:
            line = raw_line.strip()
            if line == "format ascii 1.0":
                continue
            if line.startswith("format "):
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element "):
                in_vertex = False
                continue
            if in_vertex and line.startswith("property "):
                properties.append(line.split()[-1])
                continue
            if line == "end_header":
                break
        else:
            raise ValueError(f"PLY header has no end_header: {path}")

        if vertex_count is None:
            raise ValueError(f"PLY header has no vertex count: {path}")
        if vertex_count == 0:
            return np.empty((0, 3), dtype=np.float32), None

        values = np.loadtxt(stream, max_rows=vertex_count, ndmin=2)

    if values.shape[0] != vertex_count:
        raise ValueError(
            f"Expected {vertex_count} vertices, read {values.shape[0]} from {path}"
        )
    indices = {name: index for index, name in enumerate(properties)}
    missing = [name for name in ("x", "y", "z") if name not in indices]
    if missing:
        raise ValueError(f"PLY is missing properties {missing}: {path}")
    points = values[:, [indices["x"], indices["y"], indices["z"]]]
    colors = None
    if all(name in indices for name in ("red", "green", "blue")):
        colors = values[:, [indices["red"], indices["green"], indices["blue"]]]
        colors = np.clip(colors / 255.0, 0.0, 1.0)
    return points.astype(np.float32), colors


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    """Match build_cable_point_cloud.py's equal-axis visualization."""
    if points.shape[0] == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 0.05
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def show_point_cloud(path: Path, point_size: float, use_rgb: bool) -> None:
    points, colors = load_ascii_ply(path)
    figure = plt.figure(figsize=(8, 7))
    axes = figure.add_subplot(1, 1, 1, projection="3d")
    if len(points):
        scatter_options = {"s": point_size, "depthshade": True}
        if use_rgb and colors is not None:
            scatter_options["c"] = colors
        axes.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_options)
    set_equal_3d_axes(axes, points)
    axes.set_title(f"Cable Point Cloud (camera)\n{path.name} — {len(points)} points")
    axes.set_xlabel("X [m]")
    axes.set_ylabel("Y [m]")
    axes.set_zlabel("Z [m]")
    figure.tight_layout()
    print(f"Loaded {len(points)} points from {path}")
    print("Drag to rotate; scroll to zoom; close the window to exit.")
    plt.show(block=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="ASCII cable_camera_XXXXXXXX.ply")
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Use saved RGB colors; default matches the original blue scatter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.ply.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    show_point_cloud(path, args.point_size, args.rgb)


if __name__ == "__main__":
    main()
