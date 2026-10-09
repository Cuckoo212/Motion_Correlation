"""Transport correctness without ROS, a camera, or CUDA."""
import socket
import threading
import unittest
from types import SimpleNamespace
import numpy as np
from pointcloud.rgbd_transport import (FramePairer, LatestFrame, decode_image,
                                      send_frame, receive_frame)


class TransportTests(unittest.TestCase):
    def test_socket_roundtrip_and_latest_request(self):
        from pointcloud.ros2_frame_receiver import serve
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'frames.sock')
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(path)
            listener.listen(1)
            latest, stop = LatestFrame(), threading.Event()
            rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            depth = np.array([[0, 1, 65535], [200, 400, 800]], np.uint16)
            latest.put(({'timestamp_ns': 1}, rgb, depth))
            latest.put(({'timestamp_ns': 2}, rgb, depth))
            thread = threading.Thread(target=serve, args=(listener, latest, stop))
            thread.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(2)
                    client.connect(path)
                    client.sendall(b'N')
                    metadata, actual_rgb, actual_depth = receive_frame(client)
                    self.assertEqual(metadata['timestamp_ns'], 2)
                    np.testing.assert_array_equal(rgb, actual_rgb)
                    np.testing.assert_array_equal(depth, actual_depth)
                    latest.put(({'timestamp_ns': 3}, rgb, depth))
                    latest.put(({'timestamp_ns': 4}, rgb, depth))
                    client.sendall(b'N')
                    self.assertEqual(receive_frame(client)[0]['timestamp_ns'], 4)
            finally:
                stop.set()
                thread.join(2)
                listener.close()
            self.assertFalse(thread.is_alive())

    def test_remote_output_same_frame(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from pointcloud import remote_cable_pc
        from pointcloud.ros2_frame_receiver import serve
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            path = str(root / 'remote.sock')
            listener.bind(path)
            listener.listen(1)
            latest, stop = LatestFrame(), threading.Event()
            rgb = np.full((2, 3, 3), 50, np.uint8)
            depth = np.full((2, 3), 1000, np.uint16)
            metadata = dict(timestamp_ns=123, received_timestamp_ns=456,
                            received_pair_index=7, frame_id='camera',
                            camera_info={'camera_matrix': [[100, 0, 1], [0, 100, 1], [0, 0, 1]]})
            latest.put((metadata, rgb, depth))
            thread = threading.Thread(target=serve, args=(listener, latest, stop))
            thread.start()
            handler = __import__('signal').getsignal(__import__('signal').SIGINT)
            term_handler = __import__('signal').getsignal(__import__('signal').SIGTERM)
            try:
                argv = ['remote', '--socket-path', path, '--dataset-root', str(root / 'data'),
                        '--output-root', str(root / 'output'), '--sequence-name', 'test',
                        '--no-view', '--max-frames', '1']
                with patch('sys.argv', argv), patch.object(remote_cable_pc, 'Sam3CableSegmenter') as segmenter:
                    segmenter.return_value.segment.return_value = (np.full((2, 3), 255, np.uint8), .9)
                    remote_cable_pc.main()
                    np.testing.assert_array_equal(segmenter.return_value.segment.call_args.args[0], rgb)
                data = root / 'data' / 'test'
                metadata = json.loads((data / 'frames.json').read_text())
                self.assertEqual(metadata['frames'][0]['timestamp_ns'], 123)
                self.assertEqual(metadata['sam_results'][0]['point_count'], 6)
                for name in ('rgb_00000000.png', 'depth_00000000.png', 'mask_00000000.png'):
                    self.assertTrue((data / name).exists())
                self.assertIn('element vertex 6', (root / 'output/test/cable_camera_00000000.ply').read_text())
            finally:
                import signal
                signal.signal(signal.SIGINT, handler)
                signal.signal(signal.SIGTERM, term_handler)
                stop.set()
                thread.join(2)
                listener.close()

    def test_pairing_reordering_and_bounds(self):
        pairer = FramePairer(capacity=2)
        rgb = np.zeros((2, 3, 3), np.uint8)
        depth = np.zeros((2, 3), np.uint16)
        self.assertIsNone(pairer.add('depth', 20, 'camera', depth))
        self.assertIsNotNone(pairer.add('rgb', 20, 'camera', rgb))
        pairer.add('rgb', 10, 'camera', rgb)
        self.assertIsNone(pairer.add('depth', 10, 'camera', depth))
        for stamp in range(30, 40):
            pairer.add('rgb', stamp, 'camera', rgb)
        self.assertLessEqual(len(pairer.pending), 2)
        with self.assertRaises(ValueError):
            pairer.add('depth', 39, 'wrong-camera', depth)

    def test_depth_stride_and_endianness(self):
        depth = np.array([[100, 65535], [0, 1000]], dtype='>u2')
        data = b''.join(row.tobytes() + b'\x00\x00' for row in depth)
        message = SimpleNamespace(encoding='16UC1', width=2, height=2,
                                  step=6, is_bigendian=True, data=data)
        np.testing.assert_array_equal(decode_image(message, '16UC1'), depth)
        message.step = 2
        with self.assertRaises(ValueError):
            decode_image(message, '16UC1')

    def test_truncated_packet(self):
        left, right = socket.socketpair()
        try:
            left.sendall(b'\x00\x00')
            left.close()
            with self.assertRaises(EOFError):
                receive_frame(right)
        finally:
            left.close()
            right.close()


if __name__ == '__main__':
    unittest.main()
