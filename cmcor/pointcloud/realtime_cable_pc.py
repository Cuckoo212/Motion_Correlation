#!/usr/bin/env python3
"""Record aligned RealSense RGB-D and build SAM3 cable-only point clouds."""

from __future__ import annotations

import argparse
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .geometry import depth_mask_to_points, save_ply, voxel_downsample
from .sam3_segmenter import Sam3CableSegmenter
from .runtime import Frame, FrameRecorder, MatplotlibViewer, atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "realtime_cable_pc"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "realtime_cable_pc"
DEFAULT_SAM3_PROJECT = Path(__file__).resolve().parent / "sam3_project"



def parse_args(argv: list[str] | None = None, *, remote: bool = False) -> argparse.Namespace:
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
    if remote:
        parser.description = "Receive remote RGB-D and run SAM3 cable point-cloud processing."
        parser.add_argument("--socket-path", type=Path, default=Path("/tmp/cable_rgbd.sock"))
        parser.add_argument("--receive-timeout", type=float, default=10.0)
    else:
        parser.add_argument("--width", type=int, default=640)
        parser.add_argument("--height", type=int, default=480)
        parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-depth-mm", type=int, default=150)
    parser.add_argument("--max-depth-mm", type=int, default=2000)
    parser.add_argument("--voxel-size-mm", type=float, default=2.0)
    parser.add_argument(
        "--reuse-last-mask",
        action="store_true",
        help=(
            "Save mask_DDDDDDDD.png for every RGB frame by reusing the latest "
            "SAM result. Disabled by default because moving cables can misalign."
        ),
    )
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--view-width", type=int, default=1280)
    parser.add_argument("--view-height", type=int, default=800)
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Rendered point size in the Matplotlib 3D scatter window.",
    )
    parser.add_argument(
        "--view-rgb",
        action="store_true",
        help="Show saved RGB point colors instead of the original blue scatter.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args(argv)


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
    viewer = MatplotlibViewer(
        not args.no_view,
        width=args.view_width,
        height=args.view_height,
        point_size=args.point_size,
        use_rgb=args.view_rgb,
    )

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
    last_mask: np.ndarray | None = None
    statistics: list[dict] = []

    print(f"Dataset sequence: {sequence_dir}")
    print(f"Point-cloud output: {output_dir}")
    print("Press Ctrl-C or close the Matplotlib window to stop.")
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
            save_ply(
                output_dir / f"cable_camera_{frame_index:08d}.ply",
                points,
                colors,
            )
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
            sequence_dir / "processing_stats.json",
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
