"""Run with sourced ROS environment: PointCloud2 layout and live DDS roundtrip."""
import unittest
import numpy as np
try:
    import rclpy
    from cable_pointcloud.ros_nodes import cloud_message, decode_cloud
except ImportError:
    rclpy = None


@unittest.skipIf(rclpy is None, 'ROS environment not sourced')
class CloudTests(unittest.TestCase):
    def test_empty_and_colored_roundtrip(self):
        for count in (0, 3):
            points = np.arange(count*3, dtype=np.float32).reshape(count, 3) / 10
            colors = np.arange(count*3, dtype=np.uint8).reshape(count, 3)
            message = cloud_message(dict(frame_id='camera', timestamp_ns=12345678901), points, colors)
            actual, rgb = decode_cloud(message)
            np.testing.assert_array_equal(actual, points)
            np.testing.assert_array_equal(rgb, colors)
            self.assertEqual(message.header.stamp.sec, 12)
            self.assertEqual(message.header.stamp.nanosec, 345678901)
            message.point_step = 12
            with self.assertRaises(ValueError):
                decode_cloud(message)

    def test_live_ros_roundtrip(self):
        import time
        from sensor_msgs.msg import PointCloud2
        from rclpy.qos import qos_profile_sensor_data
        rclpy.init()
        node = rclpy.create_node('cloud_roundtrip_test')
        received = []
        node.create_subscription(PointCloud2, '/cable_cloud_test', received.append, qos_profile_sensor_data)
        pub = node.create_publisher(PointCloud2, '/cable_cloud_test', qos_profile_sensor_data)
        message = cloud_message(dict(frame_id='camera', timestamp_ns=10), np.ones((2, 3)), None)
        try:
            deadline = time.monotonic() + 5
            while not received and time.monotonic() < deadline:
                pub.publish(message)
                rclpy.spin_once(node, timeout_sec=.1)
            self.assertTrue(received)
            np.testing.assert_array_equal(decode_cloud(received[0])[0], np.ones((2, 3)))
        finally:
            node.destroy_node()
            rclpy.shutdown()
