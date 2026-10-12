#!/usr/bin/env python3
"""GPU PC ROS 2 receiver; serves the newest paired frame to a local SAM3 process."""
from __future__ import annotations
import argparse
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path
from .rgbd_transport import FramePairer, LatestFrame, decode_image, send_frame


def serve(listener, latest, stop):
    listener.settimeout(0.5)
    while not stop.is_set():
        try:
            connection, _ = listener.accept()
        except socket.timeout:
            continue
        with connection:
            connection.settimeout(1.)
            version = 0
            while not stop.is_set():
                try:
                    request = connection.recv(1)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if request != b'N':
                    break
                result = latest.get_after(version, stop)
                if result is None:
                    break
                version, (metadata, rgb, depth) = result
                try:
                    send_frame(connection, metadata, rgb, depth)
                except (OSError, ValueError):
                    # Partial sends invalidate framing: close rather than resume.
                    break



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket-path', type=Path, default=Path('/tmp/cable_rgbd.sock'))
    parser.add_argument('--topic-prefix', default='/cable_camera')
    args, ros_args = parser.parse_known_args()
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import String
    path = args.socket_path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'{path} already exists; stop the other receiver or remove a confirmed stale socket')
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    os.chmod(path, 0o600)
    listener.listen(1)
    stop = threading.Event()
    latest, pairer = LatestFrame(), FramePairer()
    thread = None
    node = None
    rclpy.init(args=ros_args)
    try:
        node = rclpy.create_node('cable_rgbd_receiver')
        camera = None
        sdk_metadata = None
        received = 0
        last_received = time.monotonic()
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        prefix = args.topic_prefix.rstrip('/')
        def on_info(message):
            nonlocal camera
            camera = message
        def on_metadata(message):
            nonlocal sdk_metadata
            try:
                value = json.loads(message.data)
                if value.get('depth_unit') != 'millimetre in saved PNG' or value.get('color_aligned_depth') is not True:
                    raise ValueError('Publisher must provide aligned millimetre depth')
                sdk_metadata = value
            except (ValueError, TypeError) as exc:
                sdk_metadata = None
                node.get_logger().error(str(exc))
        def on_image(kind, message):
            nonlocal received, last_received
            if camera is None or sdk_metadata is None:
                return
            try:
                stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                result = pairer.add(kind, stamp, message.header.frame_id,
                                    decode_image(message, 'bgr8' if kind == 'rgb' else '16UC1'))
                if result is None:
                    return
                frame_id, rgb, depth = result
                if frame_id != camera.header.frame_id or depth.shape != (camera.height, camera.width):
                    raise ValueError('CameraInfo does not match RGB-D')
                received += 1
                last_received = time.monotonic()
                metadata = dict(timestamp_ns=stamp, received_timestamp_ns=time.time_ns(),
                                received_pair_index=received - 1, frame_id=frame_id,
                                camera_info=dict(sdk_metadata, camera_matrix=[list(camera.k[i:i+3]) for i in (0, 3, 6)]))
                latest.put((metadata, rgb, depth))
            except (ValueError, TypeError) as exc:
                node.get_logger().warning(f'Dropped RGB-D: {exc}')
        subscriptions = [
            node.create_subscription(CameraInfo, prefix + '/color/camera_info', on_info, info_qos),
            node.create_subscription(String, prefix + '/metadata', on_metadata, info_qos),
            node.create_subscription(Image, prefix + '/color/image_raw', lambda m: on_image('rgb', m), image_qos),
            node.create_subscription(Image, prefix + '/aligned_depth/image_raw', lambda m: on_image('depth', m), image_qos)]
        def report():
            age = time.monotonic() - last_received
            node.get_logger().info(f'Paired frames={received}; last receive {age:.1f}s ago')
            if age > 5:
                node.get_logger().warning('No fresh RGB-D: check publisher, metadata, domain ID and network')
        timer = node.create_timer(5., report)
        thread = threading.Thread(target=serve, args=(listener, latest, stop), daemon=True)
        thread.start()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        node.get_logger().info(f'Local SAM3 socket: {path}; waiting for {prefix}')
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        stop.set()
        if thread:
            thread.join(timeout=2.)
        listener.close()
        path.unlink(missing_ok=True)
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
