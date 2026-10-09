#!/usr/bin/env python3
"""Run SAM3 on remote RGB-D received by the separate ROS 2 process."""
from __future__ import annotations
import signal
import socket
import time
from datetime import datetime
import cv2
import numpy as np
from .geometry import depth_mask_to_points, voxel_downsample, save_ply
from .runtime import MatplotlibViewer, atomic_json
from .rgbd_transport import receive_frame
from .sam3_segmenter import Sam3CableSegmenter
from .realtime_cable_pc import parse_args


def main():
    args = parse_args(remote=True)
    if args.receive_timeout <= 0:
        raise ValueError('receive-timeout must be positive')
    if args.reuse_last_mask:
        raise ValueError('Remote mode always uses the mask from the same RGB-D pair')
    name = args.sequence_name or datetime.now().strftime('%Y-%m-%d-%H%M%S')
    dataset = args.dataset_root.expanduser().resolve() / name
    output = args.output_root.expanduser().resolve() / name
    if dataset.exists() or output.exists():
        raise FileExistsError(f'Sequence already exists: {name}')
    checkpoint = args.checkpoint or args.sam3_project / 'checkpoints' / 'sam3.pt'
    segmenter = Sam3CableSegmenter(args.sam3_project, checkpoint, prompt=args.prompt,
        confidence_threshold=args.confidence_threshold, min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio, selection=args.selection)
    viewer = MatplotlibViewer(not args.no_view, width=args.view_width, height=args.view_height,
                              point_size=args.point_size, use_rgb=args.view_rgb)
    stopping = False
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(args.receive_timeout)
    def stop(*_):
        nonlocal stopping
        stopping = True
        connection.close()  # Interrupt a blocked receive immediately.
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    frames, results = [], []
    created = False
    try:
        connection.connect(str(args.socket_path.expanduser()))
        dataset.mkdir(parents=True)
        output.mkdir(parents=True)
        created = True
        print(f'Dataset: {dataset}\nPoint clouds: {output}')
        while not stopping:
            connection.sendall(b'N')
            metadata, rgb, depth = receive_frame(connection)
            started = time.perf_counter()
            source_age_ms = (time.time_ns() - metadata['timestamp_ns']) / 1e6
            camera = metadata['camera_info']
            if not frames:
                atomic_json(dataset / 'camera_info.json', camera)
            mask, score = segmenter.segment(rgb)
            inference_ms = (time.perf_counter() - started) * 1000.
            points, colors = depth_mask_to_points(depth, mask, np.asarray(camera['camera_matrix']), rgb,
                min_depth_mm=args.min_depth_mm, max_depth_mm=args.max_depth_mm)
            points, colors = voxel_downsample(points, colors, args.voxel_size_mm * .001)
            index = len(frames)
            stem = f'{index:08d}'
            for filename, image in ((f'rgb_{stem}.png', rgb), (f'depth_{stem}.png', depth), (f'mask_{stem}.png', mask)):
                if not cv2.imwrite(str(dataset / filename), image):
                    raise OSError(f'Failed to save {filename}')
            save_ply(output / f'cable_camera_{stem}.ply', points, colors)
            frames.append(dict(frame_index=index, timestamp_ns=metadata['timestamp_ns'],
                received_timestamp_ns=metadata['received_timestamp_ns'],
                received_pair_index=metadata['received_pair_index'], frame_id=metadata['frame_id'],
                rgb=f'rgb_{stem}.png', depth=f'depth_{stem}.png', camera_info=camera))
            results.append(dict(frame_index=index, timestamp_ns=metadata['timestamp_ns'],
                sam_score=score, sam_inference_ms=inference_ms, point_count=len(points), mask=f'mask_{stem}.png',
                source_age_at_inference_ms=source_age_ms,
                result_age_ms=(time.time_ns() - metadata['timestamp_ns']) / 1e6))
            print(f'frame={index:08d} SAM={inference_ms:.1f}ms age={results[-1]["result_age_ms"]:.1f}ms points={len(points)}')
            if not viewer.update(points, colors) or (args.max_frames and len(frames) >= args.max_frames):
                break
    except (OSError, EOFError) as exc:
        if not stopping:
            raise RuntimeError('RGB-D connection failed or timed out; check receiver/publisher, then restart with a new sequence name') from exc
    finally:
        connection.close()
        viewer.close()
        if created:
            atomic_json(dataset / 'frames.json', {'frames': frames, 'sam_results': results})
            atomic_json(dataset / 'processing_stats.json', dict(sequence_name=name,
                captured_frames=len(frames), sam_frames=len(results), mode='remote_ros2_latest_frame',
                mean_sam_inference_ms=float(np.mean([r['sam_inference_ms'] for r in results])) if results else None,
                mean_result_age_ms=float(np.mean([r['result_age_ms'] for r in results])) if results else None))


if __name__ == '__main__':
    main()
