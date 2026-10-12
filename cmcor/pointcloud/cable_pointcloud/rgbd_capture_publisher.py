#!/usr/bin/env python3
"""Mini PC RealSense acquisition: aligned RGB-D over standard ROS 2 topics."""
from __future__ import annotations
import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from .runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--publish-fps', type=float, default=5)
    parser.add_argument('--serial')
    parser.add_argument('--topic-prefix', default='/cable_camera')
    parser.add_argument('--frame-id', default='cable_camera_color_optical_frame')
    parser.add_argument('--record-root', type=Path, help='Optional full-rate RGB-D recording root')
    parser.add_argument('--sequence-name')
    parser.add_argument('--max-frames', type=int, default=0, help='Stop after this many acquired frames')
    args, ros_args = parser.parse_known_args()
    if min(args.width, args.height, args.fps, args.publish_fps) <= 0 or args.publish_fps > args.fps:
        parser.error('Positive dimensions/rates required; publish-fps must not exceed fps')
    import pyrealsense2 as rs
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String
    rclpy.init(args=ros_args)
    node = rclpy.create_node('cable_rgbd_capture')
    image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
    info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)
    prefix = args.topic_prefix.rstrip('/')
    rgb_pub = node.create_publisher(Image, prefix + '/color/image_raw', image_qos)
    depth_pub = node.create_publisher(Image, prefix + '/aligned_depth/image_raw', image_qos)
    info_pub = node.create_publisher(CameraInfo, prefix + '/color/camera_info', info_qos)
    metadata_pub = node.create_publisher(String, prefix + '/metadata', info_qos)
    pipeline, config = rs.pipeline(), rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    started = False
    records = []
    record_dir = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        profile = pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        matrix = [[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]]
        metadata = dict(camera_matrix=matrix, distortion_coefficients=list(intr.coeffs),
                        distortion_model=str(intr.model), width=intr.width, height=intr.height,
                        fps=args.fps, publish_fps=args.publish_fps, depth_scale_m=scale,
                        depth_unit='millimetre in saved PNG', color_aligned_depth=True,
                        timestamp_source='Mini PC host Unix time immediately after frame acquisition',
                        device_serial=profile.get_device().get_info(rs.camera_info.serial_number))
        if args.record_root:
            record_dir = args.record_root.expanduser().resolve() / (args.sequence_name or datetime.now().strftime('%Y-%m-%d-%H%M%S'))
            record_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(record_dir / 'camera_info.json', metadata)
        info = CameraInfo()
        info.header.frame_id = args.frame_id
        info.width, info.height = intr.width, intr.height
        # Preserve the SDK model exactly in /metadata. ROS model mapping is explicit.
        if intr.model in (rs.distortion.none, rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
            info.distortion_model = 'plumb_bob'
            info.d = list(intr.coeffs)
        else:
            info.distortion_model = ''
            info.d = []
            node.get_logger().warning('SDK distortion model is in /metadata; no ROS distortion mapping provided')
        info.k = np.asarray(matrix).ravel().tolist()
        info.r = np.eye(3).ravel().tolist()
        info.p = [intr.fx, 0., intr.ppx, 0., 0., intr.fy, intr.ppy, 0., 0., 0., 1., 0.]
        metadata_pub.publish(String(data=json.dumps(metadata)))
        info_pub.publish(info)
        next_publish = 0.
        index = 0
        node.get_logger().info(f'Publishing aligned millimetre RGB-D at {args.publish_fps} Hz on {prefix}')
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=0)
            frames = align.process(pipeline.wait_for_frames(timeout_ms=1000))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            timestamp = time.time_ns()
            rgb = np.asanyarray(color.get_data()).copy()
            depth_mm = np.clip(np.asanyarray(depth.get_data()).astype(np.float64) * scale * 1000., 0, 65535).astype(np.uint16)
            if record_dir:
                stem = f'{index:08d}'
                for name, array in ((f'rgb_{stem}.png', rgb), (f'depth_{stem}.png', depth_mm)):
                    if not cv2.imwrite(str(record_dir / name), array):
                        raise OSError(f'Failed to save {name}')
                records.append(dict(frame_index=index, timestamp_ns=timestamp,
                                    source_frame_number=int(color.get_frame_number()),
                                    camera_timestamp_ms=float(color.get_timestamp()),
                                    rgb=f'rgb_{stem}.png', depth=f'depth_{stem}.png'))
            now = time.monotonic()
            if now >= next_publish:
                for publisher, array, encoding, step in (
                    (rgb_pub, rgb, 'bgr8', intr.width * 3),
                    (depth_pub, depth_mm.astype('<u2'), '16UC1', intr.width * 2)):
                    message = Image()
                    message.header.frame_id = args.frame_id
                    message.header.stamp.sec = timestamp // 1_000_000_000
                    message.header.stamp.nanosec = timestamp % 1_000_000_000
                    message.width, message.height = intr.width, intr.height
                    message.encoding, message.step = encoding, step
                    message.is_bigendian = 0
                    message.data = array.tobytes()
                    publisher.publish(message)
                next_publish = now + 1. / args.publish_fps
            index += 1
            if args.max_frames and index >= args.max_frames:
                break
    finally:
        if started:
            pipeline.stop()
        if record_dir:
            atomic_json(record_dir / 'frames.json', {'frames': records})
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
