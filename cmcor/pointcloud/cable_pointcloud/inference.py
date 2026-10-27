#!/usr/bin/env python3
"""Compute and record clouds from the inference node's in-memory RGB-D input."""
from __future__ import annotations
import signal
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from .geometry import depth_mask_to_points, voxel_downsample, save_ply
from .runtime import MatplotlibViewer, atomic_json
from .sam3_segmenter import Sam3CableSegmenter
from .inference_args import parse_args


class RunStats:
    """Constant-memory cumulative statistics, independent of recording."""
    def __init__(self):
        self.count = 0
        self.sam_ms = self.age_ms = 0.
        self.first_frame = self.last_frame = None
        self.first_done = self.last_done = None

    def add(self, frame, result):
        if not self.count:
            self.first_frame = frame
            self.first_done = result['completed_monotonic_s']
        self.last_frame = frame
        self.last_done = result['completed_monotonic_s']
        self.count += 1
        self.sam_ms += result['sam_inference_ms']
        self.age_ms += result['result_age_ms']

    def summary(self):
        if not self.count:
            return {}
        first, last = self.first_frame, self.last_frame
        receive_seconds = (last['received_timestamp_ns'] - first['received_timestamp_ns']) / 1e9
        completion_seconds = self.last_done - self.first_done
        pairs = last['received_pair_index'] - first['received_pair_index']
        return dict(capture_config_hz=first['camera_info'].get('fps'),
                    publish_config_hz=first['camera_info'].get('publish_fps'),
                    receiver_hz=pairs / receive_seconds if receive_seconds > 0 else None,
                    cloud_hz=(self.count - 1) / completion_seconds if completion_seconds > 0 else None,
                    mean_sam_inference_ms=self.sam_ms / self.count,
                    skipped_pairs=max(0, pairs + 1 - self.count))


def print_rates(summary, *, final=False):
    def fmt(value):
        return 'N/A' if value is None else f'{value:.2f}'
    print(f"[rates{' final' if final else ''}] "
          f"capture_config={fmt(summary['capture_config_hz'])}Hz "
          f"publish_config={fmt(summary['publish_config_hz'])}Hz "
          f"receiver={fmt(summary['receiver_hz'])}Hz "
          f"cloud={fmt(summary['cloud_hz'])}Hz "
          f"SAM_mean={fmt(summary['mean_sam_inference_ms'])}ms "
          f"skipped={summary['skipped_pairs']} (run average)", flush=True)


def main(source, on_cloud=None, stop_event=None):
    args = parse_args()
    if args.receive_timeout <= 0:
        raise ValueError('receive-timeout must be positive')
    name = args.sequence_name or datetime.now().strftime('%Y-%m-%d-%H%M%S')
    dataset = args.dataset_root.expanduser().resolve() / name
    output = args.output_root.expanduser().resolve() / name
    if args.record and (dataset.exists() or output.exists()):
        raise FileExistsError(f'Sequence already exists: {name}')
    checkpoint = args.checkpoint or args.sam3_project / 'checkpoints' / 'sam3.pt'
    segmenter = Sam3CableSegmenter(args.sam3_project, checkpoint, prompt=args.prompt,
        confidence_threshold=args.confidence_threshold, min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio, selection=args.selection)
    viewer = MatplotlibViewer(not args.no_view, width=args.view_width, height=args.view_height,
                              point_size=args.point_size, use_rgb=args.view_rgb)
    stop_event = stop_event or threading.Event()
    def stop(*_):
        stop_event.set()
    previous_int = signal.signal(signal.SIGINT, stop)
    previous_term = signal.signal(signal.SIGTERM, stop)
    version = 0
    frames, results = [], []
    stats = RunStats()
    created = False
    next_rate_log = time.monotonic() + 5.
    previous_publish = None
    try:
        if args.record:
            dataset.mkdir(parents=True)
            output.mkdir(parents=True)
            created = True
            print(f'Recording ON\nDataset: {dataset}\nPoint clouds: {output}')
        else:
            print('Recording OFF: compute and publish only; no dataset/output files', flush=True)
        while not stop_event.is_set():
            wait_started = time.monotonic()
            frame = source.get_after(version, stop_event, args.receive_timeout)
            input_wait_ms = (time.monotonic() - wait_started) * 1000.
            if frame is None:
                break
            version, (metadata, rgb, depth) = frame
            started = time.perf_counter()
            source_age_ms = (time.time_ns() - metadata['timestamp_ns']) / 1e6
            camera = metadata['camera_info']
            if not stats.count:
                if args.record:
                    atomic_json(dataset / 'camera_info.json', camera)
                print(f"[config] capture={camera.get('fps', 'N/A')}Hz "
                      f"publish_limit={camera.get('publish_fps', 'N/A')}Hz; "
                      "actual rates logged every 5s", flush=True)
            mask, score = segmenter.segment(rgb)
            inference_ms = (time.perf_counter() - started) * 1000.
            geometry_started = time.monotonic()
            points, colors = depth_mask_to_points(depth, mask, np.asarray(camera['camera_matrix']), rgb,
                min_depth_mm=args.min_depth_mm, max_depth_mm=args.max_depth_mm)
            points, colors = voxel_downsample(points, colors, args.voxel_size_mm * .001)
            geometry_ms = (time.monotonic() - geometry_started) * 1000.
            index = stats.count
            stem = f'{index:08d}'
            if args.record:
                for filename, image in ((f'rgb_{stem}.png', rgb), (f'depth_{stem}.png', depth), (f'mask_{stem}.png', mask)):
                    if not cv2.imwrite(str(dataset / filename), image):
                        raise OSError(f'Failed to save {filename}')
                save_ply(output / f'cable_camera_{stem}.ply', points, colors)
            publish_started = time.monotonic()
            if on_cloud is not None:
                on_cloud(metadata, points, colors)
            publish_done = time.monotonic()
            publish_ms = (publish_done - publish_started) * 1000.
            publish_gap_ms = ((publish_done - previous_publish) * 1000.
                              if previous_publish is not None else None)
            previous_publish = publish_done
            frame_record = dict(frame_index=index, timestamp_ns=metadata['timestamp_ns'],
                received_timestamp_ns=metadata['received_timestamp_ns'],
                received_pair_index=metadata['received_pair_index'], frame_id=metadata['frame_id'],
                rgb=f'rgb_{stem}.png', depth=f'depth_{stem}.png', camera_info=camera)
            result_record = dict(frame_index=index, timestamp_ns=metadata['timestamp_ns'],
                sam_score=score, sam_inference_ms=inference_ms, point_count=len(points), mask=f'mask_{stem}.png',
                input_wait_ms=input_wait_ms, geometry_ms=geometry_ms, publish_ms=publish_ms,
                publish_done_monotonic_s=publish_done, publish_gap_ms=publish_gap_ms,
                source_age_at_inference_ms=source_age_ms,
                completed_monotonic_s=time.monotonic(),
                result_age_ms=(time.time_ns() - metadata['timestamp_ns']) / 1e6)
            stats.add(frame_record, result_record)
            if args.record:
                frames.append(frame_record)
                results.append(result_record)
            gap = 'N/A' if publish_gap_ms is None else f'{publish_gap_ms:.1f}ms'
            print(f'frame={index:08d} SAM={inference_ms:.1f}ms '
                  f'age={result_record["result_age_ms"]:.1f}ms points={len(points)} '
                  f'wait={input_wait_ms:.1f}ms geometry={geometry_ms:.1f}ms '
                  f'publish={publish_ms:.1f}ms publish_gap={gap} '
                  f'done_monotonic={publish_done:.6f}', flush=True)
            view_open = viewer.update(points, colors)
            if time.monotonic() >= next_rate_log:
                print_rates(stats.summary())
                next_rate_log = time.monotonic() + 5.
            if not view_open or (args.max_frames and stats.count >= args.max_frames):
                break
    except (OSError, EOFError) as exc:
        if not stop_event.is_set():
            raise RuntimeError('RGB-D input failed or timed out; check Mini PC publisher, then restart with a new sequence name') from exc
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        viewer.close()
        rates = stats.summary()
        if rates:
            print_rates(rates, final=True)
        if created:
            atomic_json(dataset / 'frames.json', {'frames': frames, 'sam_results': results})
            atomic_json(dataset / 'processing_stats.json', dict(sequence_name=name,
                captured_frames=len(frames), sam_frames=len(results), mode='remote_ros2_latest_frame',
                rates=rates,
                mean_sam_inference_ms=stats.sam_ms / stats.count if stats.count else None,
                mean_result_age_ms=stats.age_ms / stats.count if stats.count else None))

