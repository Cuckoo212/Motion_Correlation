"""Shared recording and visualization utilities; no camera or SAM3 dependency."""
from __future__ import annotations
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_ns: int
    rgb_bgr: np.ndarray
    depth_mm: np.ndarray


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


class FrameRecorder(threading.Thread):
    def __init__(
        self,
        sequence_dir: Path,
        frame_queue: "queue.Queue[Frame | None]",
    ) -> None:
        super().__init__(name="rgbd-recorder", daemon=False)
        self.sequence_dir = sequence_dir
        self.frame_queue = frame_queue
        self.error: Exception | None = None
        self.frame_metadata: list[dict] = []

    def run(self) -> None:
        try:
            while True:
                frame = self.frame_queue.get()
                if frame is None:
                    return
                stem = f"{frame.index:08d}"
                rgb_path = self.sequence_dir / f"rgb_{stem}.png"
                depth_path = self.sequence_dir / f"depth_{stem}.png"
                if not cv2.imwrite(str(rgb_path), frame.rgb_bgr):
                    raise OSError(f"Failed to save {rgb_path}")
                if not cv2.imwrite(str(depth_path), frame.depth_mm):
                    raise OSError(f"Failed to save {depth_path}")
                self.frame_metadata.append(
                    {
                        "frame_index": frame.index,
                        "timestamp_ns": frame.timestamp_ns,
                        "rgb": rgb_path.name,
                        "depth": depth_path.name,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            self.error = exc


class MatplotlibViewer:
    """Live Matplotlib 3D scatter matching view_saved_cable_ply's interface."""

    def __init__(
        self,
        enabled: bool,
        *,
        width: int = 1280,
        height: int = 800,
        point_size: float = 8.0,
        use_rgb: bool = False,
    ) -> None:
        self.enabled = enabled
        self.closed = False
        if not enabled:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Matplotlib visualization requested but matplotlib is not installed. "
                "Install it or run with --no-view."
            ) from exc
        self.plt = plt
        self.point_size = point_size
        self.use_rgb = use_rgb
        plt.ion()
        self.figure = plt.figure(
            "Realtime SAM3 Cable Point Cloud",
            figsize=(width / 100.0, height / 100.0),
            dpi=100,
        )
        self.axes = self.figure.add_subplot(1, 1, 1, projection="3d")
        self.scatter = self.axes.scatter([], [], [], s=point_size, depthshade=True)
        self.axes.set_xlabel("X [m]")
        self.axes.set_ylabel("Y [m]")
        self.axes.set_zlabel("Z [m]")
        self.axes.set_title("Cable Point Cloud (camera) — waiting for first frame")
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.figure.tight_layout()
        plt.show(block=False)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def _on_close(self, _event: object) -> None:
        self.closed = True

    def _set_equal_axes(self, points: np.ndarray) -> None:
        if not len(points):
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        centers = 0.5 * (mins + maxs)
        radius = 0.5 * float(np.max(maxs - mins))
        if radius <= 0:
            radius = 0.05
        self.axes.set_xlim(centers[0] - radius, centers[0] + radius)
        self.axes.set_ylim(centers[1] - radius, centers[1] + radius)
        self.axes.set_zlim(centers[2] - radius, centers[2] + radius)

    def update(self, points: np.ndarray, colors: np.ndarray | None) -> bool:
        if not self.enabled:
            return True
        if self.closed or not self.plt.fignum_exists(self.figure.number):
            return False
        self.scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        if self.use_rgb and colors is not None and len(colors) == len(points):
            rgb = colors.astype(np.float64) / 255.0
            self.scatter.set_facecolor(rgb)
            self.scatter.set_edgecolor(rgb)
        else:
            default_blue = np.array([[31.0 / 255.0, 119.0 / 255.0, 180.0 / 255.0, 1.0]])
            self.scatter.set_facecolor(default_blue)
            self.scatter.set_edgecolor(default_blue)
        self._set_equal_axes(points)
        self.axes.set_title(f"Cable Point Cloud (camera) — {len(points)} points")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        return not self.closed

    def close(self) -> None:
        if self.enabled and not self.closed:
            self.plt.close(self.figure)
            self.closed = True

