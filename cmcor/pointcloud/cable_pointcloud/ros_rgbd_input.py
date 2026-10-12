"""RGB-D subscriptions owned by the inference node; latest frame stays in memory."""
import json
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from .rgbd_transport import FramePairer, LatestFrame, decode_image
from .throughput import Throughput


class RosRgbdInput:
    def __init__(self, node, topic_prefix='/cable_camera'):
        self.latest = LatestFrame()
        latest, pairer = self.latest, FramePairer()
        camera = None
        sdk_metadata = None
        received = 0
        throughput = Throughput(('rgb', 'depth', 'paired', 'metadata_wait', 'invalid'))
        source_stats = None
        source_stats_time = None
        last_received = time.monotonic()
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        prefix = topic_prefix.rstrip('/')
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
            throughput.add(kind)
            if camera is None or sdk_metadata is None:
                throughput.add('metadata_wait')
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
                throughput.add('paired')
                last_received = time.monotonic()
                metadata = dict(timestamp_ns=stamp, received_timestamp_ns=time.time_ns(),
                                received_pair_index=received - 1, frame_id=frame_id,
                                camera_info=dict(sdk_metadata, camera_matrix=[list(camera.k[i:i+3]) for i in (0, 3, 6)]))
                latest.put((metadata, rgb, depth))
            except (ValueError, TypeError) as exc:
                throughput.add('invalid')
                node.get_logger().warning(f'Dropped RGB-D: {exc}')
        def on_stats(message):
            nonlocal source_stats, source_stats_time
            try:
                stats = json.loads(message.data)
                # Read only the numeric fields used in logs; reject malformed telemetry.
                values = [stats[group][key] for group in ('hz', 'counts')
                          for key in ('captured', 'published')]
                if not all(isinstance(v, (int, float)) and 0 <= v < 1e20 for v in values):
                    raise ValueError('Invalid source counters')
                source_stats, source_stats_time = stats, time.monotonic()
            except (ValueError, TypeError, KeyError) as exc:
                node.get_logger().warning(f'Invalid transport stats: {exc}')
        self.subscriptions = [
            node.create_subscription(CameraInfo, prefix + '/color/camera_info', on_info, info_qos),
            node.create_subscription(String, prefix + '/metadata', on_metadata, info_qos),
            node.create_subscription(String, prefix + '/transport_stats', on_stats, info_qos),
            node.create_subscription(Image, prefix + '/color/image_raw', lambda m: on_image('rgb', m), image_qos),
            node.create_subscription(Image, prefix + '/aligned_depth/image_raw', lambda m: on_image('depth', m), image_qos)]
        def report():
            stats = throughput.snapshot()
            counts, hz = stats['counts'], stats['hz']
            age = time.monotonic() - last_received
            node.get_logger().info(
                f"[transport GPU] rgb={hz['rgb']:.2f}Hz depth={hz['depth']:.2f}Hz "
                f"paired={hz['paired']:.2f}Hz rgb_total={counts['rgb']} depth_total={counts['depth']} "
                f"paired_total={counts['paired']} metadata_wait_total={counts['metadata_wait']} "
                f"invalid_total={counts['invalid']} pending={len(pairer.pending)} "
                f"window={stats['window_seconds']:.2f}s last_pair_age={age:.1f}s")
            if source_stats is not None:
                source_age = time.monotonic() - source_stats_time
                node.get_logger().info(
                    f"[transport MiniPC] capture={source_stats['hz']['captured']:.2f}Hz "
                    f"publish={source_stats['hz']['published']:.2f}Hz "
                    f"captured_total={source_stats['counts']['captured']} "
                    f"published_total={source_stats['counts']['published']} "
                    f"stats_age={source_age:.1f}s session={source_stats.get('session_id', 'unknown')} "
                    f"{'STALE' if source_age > 12 else ''}")
            else:
                node.get_logger().info('[transport MiniPC] N/A: waiting for /transport_stats; update Mini PC package')
            if age > 5:
                node.get_logger().warning('No fresh RGB-D: check publisher, metadata, domain ID and network')
        self.timer = node.create_timer(5., report)
        self.report = report

    def get_after(self, version, stop, timeout):
        return self.latest.get_after(version, stop, timeout)
