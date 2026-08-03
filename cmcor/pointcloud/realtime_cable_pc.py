#!/usr/bin/env python3
"""Record aligned RealSense RGB-D and build SAM3 cable-only point clouds."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .geometry import depth_mask_to_points, save_ply, voxel_downsample
from .sam3_segmenter import Sam3CableSegmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "realtime_cable_pc"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "realtime_cable_pc"
DEFAULT_SAM3_PROJECT = Path(__file__).resolve().parent / "sam3_project"


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


class Open3DViewer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.visualizer = None
        self.cloud = None
        if not enabled:
            return
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "Open3D visualization requested but open3d is not installed. "
                "Install it or run with --no-view."
            ) from exc
        self.o3d = o3d
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window("Realtime SAM3 Cable Point Cloud")
        self.cloud = o3d.geometry.PointCloud()
        self.visualizer.add_geometry(self.cloud)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.visualizer.add_geometry(coordinate)

    def update(self, points: np.ndarray, colors: np.ndarray | None) -> bool:
        if not self.enabled:
            return True
        self.cloud.points = self.o3d.utility.Vector3dVector(points)
        if colors is not None:
            self.cloud.colors = self.o3d.utility.Vector3dVector(
                colors.astype(np.float64) / 255.0
            )
        self.visualizer.update_geometry(self.cloud)
        alive = self.visualizer.poll_events()
        self.visualizer.update_renderer()
        return bool(alive)

    def close(self) -> None:
        if self.visualizer is not None:
            self.visualizer.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture aligned RealSense RGB-D, run SAM3 cable segmentation, "
            "and display/save the cable-only point cloud."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequence-name", help="Default: current local timestamp")
    parser.add_argument("--sam3-project", type=Path, default=DEFAULT_SAM3_PROJECT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="cable")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=0.0001)
    parser.add_argument("--max-area-ratio", type=float, default=0.5)
    parser.add_argument("--selection", choices=("union", "best"), default="union")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-depth-mm", type=int, default=150)
    parser.add_argument("--max-depth-mm", type=int, default=2000)
    parser.add_argument("--voxel-size-mm", type=float, default=2.0)
    parser.add_argument(
        "--save-cloud-every",
        type=int,
        default=0,
        help="Save one binary PLY every N SAM results; 0 saves only latest.ply.",
    )
    parser.add_argument(
        "--reuse-last-mask",
        action="store_true",
        help=(
            "Save mask_DDDDDDDD.png for every RGB frame by reusing the latest "
            "SAM result. Disabled by default because moving cables can misalign."
        ),
    )
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for live capture") from exc

    sequence_name = args.sequence_name or datetime.now().strftime("%Y-%m-%d-%H%M%S")
    sequence_dir = args.dataset_root.expanduser().resolve() / sequence_name
    output_dir = args.output_root.expanduser().resolve() / sequence_name
    if sequence_dir.exists() or output_dir.exists():
        raise FileExistsError(f"Sequence already exists: {sequence_name}")
    sequence_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else args.sam3_project.expanduser().resolve() / "checkpoints" / "sam3.pt"
    )
    segmenter = Sam3CableSegmenter(
        args.sam3_project,
        checkpoint,
        prompt=args.prompt,
        confidence_threshold=args.confidence_threshold,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        selection=args.selection,
    )
    viewer = Open3DViewer(not args.no_view)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.width,
        args.height,
        rs.format.z16,
        args.fps,
    )
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    intrinsics = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    camera_info = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": list(intrinsics.coeffs),
        "distortion_model": str(intrinsics.model),
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fps": args.fps,
        "depth_scale_m": depth_scale,
        "depth_unit": "millimetre in saved PNG",
        "color_aligned_depth": True,
    }
    atomic_json(sequence_dir / "camera_info.json", camera_info)

    recording_queue: "queue.Queue[Frame | None]" = queue.Queue(maxsize=args.fps * 10)
    recorder = FrameRecorder(sequence_dir, recording_queue)
    recorder.start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    frame_index = 0
    inference_count = 0
    last_mask: np.ndarray | None = None
    statistics: list[dict] = []

    print(f"Dataset sequence: {sequence_dir}")
    print(f"Point-cloud output: {output_dir}")
    print("Press Ctrl-C or close the Open3D window to stop.")
    try:
        while not stop.is_set():
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            rgb_bgr = np.asanyarray(color_frame.get_data()).copy()
            raw_depth = np.asanyarray(depth_frame.get_data())
            depth_mm = np.clip(
                raw_depth.astype(np.float64) * depth_scale * 1000.0,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            timestamp_ns = time.time_ns()
            frame = Frame(frame_index, timestamp_ns, rgb_bgr, depth_mm)
            try:
                recording_queue.put(frame, timeout=2.0)
            except queue.Full as exc:
                raise RuntimeError(
                    "RGB-D recorder is slower than capture; refusing to drop frames"
                ) from exc

            start = time.perf_counter()
            mask, score = segmenter.segment(rgb_bgr)
            inference_ms = (time.perf_counter() - start) * 1000.0
            last_mask = mask
            mask_path = sequence_dir / f"mask_{frame_index:08d}.png"
            cv2.imwrite(str(mask_path), mask)

            points, colors = depth_mask_to_points(
                depth_mm,
                mask,
                camera_matrix,
                rgb_bgr,
                min_depth_mm=args.min_depth_mm,
                max_depth_mm=args.max_depth_mm,
            )
            points, colors = voxel_downsample(
                points,
                colors,
                args.voxel_size_mm * 0.001,
            )
            save_ply(output_dir / "latest.ply", points, colors)
            if (
                args.save_cloud_every > 0
                and inference_count % args.save_cloud_every == 0
            ):
                save_ply(
                    output_dir / f"cable_{frame_index:08d}.ply",
                    points,
                    colors,
                )
            overlay = rgb_bgr.copy()
            overlay[mask > 0] = (
                overlay[mask > 0].astype(np.float32) * 0.55
                + np.array([0.0, 0.0, 255.0]) * 0.45
            ).astype(np.uint8)
            cv2.imwrite(str(output_dir / "latest_overlay.png"), overlay)
            statistics.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": timestamp_ns,
                    "sam_score": score,
                    "sam_inference_ms": inference_ms,
                    "point_count": int(len(points)),
                    "mask": mask_path.name,
                }
            )
            print(
                f"\rframe={frame_index:08d} score={score:.3f} "
                f"SAM={inference_ms:.1f} ms cloud={len(points)} points",
                end="",
                flush=True,
            )
            if not viewer.update(points, colors):
                stop.set()
            inference_count += 1
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                stop.set()
    finally:
        print()
        pipeline.stop()
        recording_queue.put(None)
        recorder.join()
        viewer.close()
        if recorder.error:
            raise recorder.error
        if args.reuse_last_mask and last_mask is not None:
            existing = {path.name for path in sequence_dir.glob("mask_*.png")}
            for item in recorder.frame_metadata:
                mask_name = f"mask_{item['frame_index']:08d}.png"
                if mask_name not in existing:
                    cv2.imwrite(str(sequence_dir / mask_name), last_mask)
        atomic_json(
            sequence_dir / "frames.json",
            {
                "frames": recorder.frame_metadata,
                "sam_results": statistics,
            },
        )
        atomic_json(
            output_dir / "processing_stats.json",
            {
                "sequence_name": sequence_name,
                "captured_frames": len(recorder.frame_metadata),
                "sam_frames": len(statistics),
                "mean_sam_inference_ms": (
                    float(np.mean([item["sam_inference_ms"] for item in statistics]))
                    if statistics
                    else None
                ),
            },
        )
        print(f"Saved {len(recorder.frame_metadata)} RGB-D frames")
        print(f"Saved {len(statistics)} SAM3 masks")


if __name__ == "__main__":
    main()
