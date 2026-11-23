"""Verify mask results are live-only across RViz subscriber restarts."""
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from cable_pointcloud.cmcor_recorder import Recorder


class MaskCloudQosTests(unittest.TestCase):
    def test_new_subscribers_receive_only_new_mask_results(self):
        from cmcor import motion_perception
        rclpy.init()
        recorder = reader = executor = None
        try:
            with patch.object(motion_perception, 'MotionPerception'):
                recorder = Recorder()
            reader = Node('cmcor_qos_reader_' + uuid4().hex[:8])
            executor = SingleThreadedExecutor()
            executor.add_node(recorder)
            executor.add_node(reader)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            def spin(seconds):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    executor.spin_once(timeout_sec=.02)
            old = PointCloud2(); old.header.frame_id = 'old'
            recorder.raw_pub.publish(old); recorder.edited_pub.publish(old)
            spin(.3)
            received_raw, received_edited = [], []
            subs = [reader.create_subscription(PointCloud2, '/cable_camera/cmcor_points_raw', received_raw.append, qos),
                    reader.create_subscription(PointCloud2, '/cable_camera/cmcor_points_edited', received_edited.append, qos)]
            spin(.8)
            self.assertEqual(received_raw, [])
            self.assertEqual(received_edited, [])
            current = PointCloud2(); current.header.frame_id = 'current'
            recorder.raw_pub.publish(current); recorder.edited_pub.publish(current)
            spin(.8)
            self.assertEqual([m.header.frame_id for m in received_raw], ['current'])
            self.assertEqual([m.header.frame_id for m in received_edited], ['current'])
            self.assertEqual(len(subs), 2)
        finally:
            if executor is not None: executor.shutdown()
            if reader is not None: reader.destroy_node()
            if recorder is not None: recorder.destroy_node()
            if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
