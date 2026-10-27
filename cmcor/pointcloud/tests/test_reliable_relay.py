"""Check the real relay delivers an unchanged cloud to a reliable reader."""
import os
import signal
import subprocess
import sys
import time
import unittest
import numpy as np
try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from cable_pointcloud.ros_nodes import cloud_message
except ImportError:
    rclpy = None


@unittest.skipIf(rclpy is None, 'ROS environment not sourced')
class ReliableRelayTests(unittest.TestCase):
    def test_reliable_reader_receives_large_cloud_unchanged(self):
        suffix = str(os.getpid())
        input_topic = '/relay_test_' + suffix + '/input'
        output_topic = '/relay_test_' + suffix + '/output'
        process = subprocess.Popen([sys.executable, '-c',
            'from cable_pointcloud.ros_nodes import publisher_main; publisher_main()',
            '--ros-args', '-p', 'input_topic:=' + input_topic,
            '-p', 'output_topic:=' + output_topic], stdout=subprocess.DEVNULL)
        rclpy.init()
        node = rclpy.create_node('reliable_relay_test_' + suffix)
        received = []
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        sub = node.create_subscription(PointCloud2, output_topic, received.append, qos)
        pub = node.create_publisher(PointCloud2, input_topic, qos_profile_sensor_data)
        cloud = cloud_message(dict(frame_id='camera', timestamp_ns=123456789),
                              np.ones((8192, 3), dtype=np.float32), None)
        try:
            deadline = time.monotonic() + 12
            while not received and time.monotonic() < deadline:
                self.assertIsNone(process.poll(), 'Relay exited before delivery')
                pub.publish(cloud)
                rclpy.spin_once(node, timeout_sec=.1)
            self.assertTrue(received, 'Reliable reader did not receive relay output')
            self.assertEqual(received[0].width, cloud.width)
            self.assertEqual(bytes(received[0].data), bytes(cloud.data))
            self.assertEqual(received[0].header, cloud.header)
        finally:
            node.destroy_node()
            rclpy.shutdown()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
