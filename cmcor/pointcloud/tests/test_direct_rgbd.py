"""Live ROS image subscriptions remain active independently of inference."""
import json
import threading
import time
import unittest
import numpy as np
try:
    import rclpy
    from cable_pointcloud.ros_rgbd_input import RosRgbdInput
except ImportError:
    rclpy = None


@unittest.skipIf(rclpy is None, 'ROS environment not sourced')
class DirectInputTests(unittest.TestCase):
    def test_direct_pairing_latest_cache_and_shutdown(self):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, CameraInfo
        from std_msgs.msg import String
        rclpy.init()
        node = rclpy.create_node('direct_rgbd_test')
        source = RosRgbdInput(node, '/test_direct_camera')
        info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        info_pub = node.create_publisher(CameraInfo, '/test_direct_camera/color/camera_info', info_qos)
        meta_pub = node.create_publisher(String, '/test_direct_camera/metadata', info_qos)
        rgb_pub = node.create_publisher(Image, '/test_direct_camera/color/image_raw', image_qos)
        depth_pub = node.create_publisher(Image, '/test_direct_camera/aligned_depth/image_raw', image_qos)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        stopping = threading.Event()
        def receive():
            while not stopping.is_set():
                executor.spin_once(timeout_sec=.02)
        thread = threading.Thread(target=receive)
        thread.start()
        def wait_for(predicate):
            deadline = time.monotonic() + 5
            while not predicate() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(predicate())
        def image(publisher, stamp, depth=False):
            message = Image()
            message.header.frame_id = 'camera'
            message.header.stamp.nanosec = stamp
            message.height, message.width = 2, 3
            message.encoding = '16UC1' if depth else 'bgr8'
            message.step = 6 if depth else 9
            array = np.full((2, 3), 1000, '<u2') if depth else np.full((2, 3, 3), stamp, np.uint8)
            message.data = array.tobytes()
            publisher.publish(message)
        try:
            wait_for(lambda: rgb_pub.get_subscription_count() > 0 and depth_pub.get_subscription_count() > 0)
            info = CameraInfo()
            info.header.frame_id = 'camera'
            info.height, info.width = 2, 3
            info.k = [100., 0., 1., 0., 100., 1., 0., 0., 1.]
            info_pub.publish(info)
            meta_pub.publish(String(data=json.dumps(dict(depth_unit='millimetre in saved PNG', color_aligned_depth=True))))
            time.sleep(.15)
            image(rgb_pub, 1)
            image(depth_pub, 2, True)
            time.sleep(.1)
            self.assertEqual(source.latest.version, 0)  # Different timestamps never pair.
            image(depth_pub, 1, True)
            wait_for(lambda: source.latest.version == 1)
            # Simulate slow inference: no consumer reads while reception advances.
            image(rgb_pub, 2)
            wait_for(lambda: source.latest.version == 2)
            version, (metadata, rgb, depth) = source.get_after(0, stopping, 1.)
            self.assertEqual(version, 2)
            self.assertEqual(metadata['timestamp_ns'], 2)
            np.testing.assert_array_equal(rgb, np.full((2, 3, 3), 2, np.uint8))
            np.testing.assert_array_equal(depth, np.full((2, 3), 1000, np.uint16))
            with self.assertRaises(TimeoutError):
                source.get_after(version, stopping, .02)
            stopping.set()
            self.assertIsNone(source.get_after(version, stopping, 1.))
        finally:
            stopping.set()
            thread.join(2)
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()
