import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from cable_pointcloud.cmcor_recorder import Recorder
from cable_pointcloud.ros_nodes import decode_cloud


class FakeMotion:
    def __init__(self):
        self.clear_buffer()
    def clear_buffer(self):
        self.image_buffer = []
        self.calls = []
    def add_sample(self, rgb, mask, point, action, depth_image):
        self.image_buffer.append({'rgb_image': rgb, 'action': action})
    def compute_correlation_images(self, depth, focal):
        masked = 'foreground_mask' in self.image_buffer[-1]
        self.calls.append((depth, focal, masked))
        self.action2index = {0: 0, 1: 1}
        # Distinguish a true second computation from just post-intersecting raw.
        self.corr_imgs = ([np.array([[0., .95], [0., 0.]]), np.zeros((2, 2))] if masked else
                          [np.array([[.95, 0.], [0., 0.]]), np.array([[0., 0.], [0., .96]])])


class Harness:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.mp = FakeMotion()
        self.messages = []
        self.clouds = {'legacy': [], 'raw': [], 'edited': []}
        self.computing = True
        self.lock = threading.Lock()
        self.pub = SimpleNamespace(publish=self.clouds['legacy'].append)
        self.raw_pub = SimpleNamespace(publish=self.clouds['raw'].append)
        self.edited_pub = SimpleNamespace(publish=self.clouds['edited'].append)
        self.foreground = np.array([[False, True], [False, False]])
    def get_parameter(self, key):
        return SimpleNamespace(value={'max_state_age': .1, 'min_frames_per_action': 2, 'mask_threshold': .86, 'output_root': self.temp.name}[key])
    def save_sequence(self, matched):
        return Recorder.save_sequence(self, matched)
    def result_mask(self, mp):
        return Recorder.result_mask(self, mp)
    def publish_result(self, target, mask, publishers, color):
        return Recorder.publish_result(self, target, mask, publishers, color)
    def request_target_mask(self, target):
        self.requested_target = target
        if isinstance(self.foreground, Exception):
            raise self.foreground
        return self.foreground
    def get_logger(self):
        return SimpleNamespace(error=lambda x: self.messages.append(x))
    def report(self, text):
        self.messages.append(text)


class RecorderTest(unittest.TestCase):
    def data(self):
        k = np.array([[100., 0, 0], [0, 100., 0], [0, 0, 1.]])
        rgb = np.full((2, 2, 3), [1, 2, 3], np.uint8)
        depth = np.full((2, 2), 1000, np.uint16)
        frames = [(1_000_000_000 + i*50_000_000, 'camera', rgb, depth, k) for i in range(4)]
        states = [(f[0], i//2, np.array([0., 0., 1.])) for i, f in enumerate(frames)]
        return frames, states
    def test_raw_and_foreground_are_separate_computations_and_outputs(self):
        frames, states = self.data()
        h = Harness(); self.addCleanup(h.temp.cleanup)
        Recorder.compute(h, frames, states)
        self.assertEqual(h.mp.calls, [(1., 100., False), (1., 100., True)])
        self.assertEqual(h.requested_target[0], frames[-1][0])
        raw, edited = h.clouds['raw'][0], h.clouds['edited'][0]
        self.assertEqual((raw.width, edited.width), (2, 1))
        self.assertEqual(raw.header, edited.header)
        self.assertEqual(raw.header.frame_id, 'camera')
        self.assertEqual(h.clouds['legacy'][0], raw)
        np.testing.assert_array_equal(decode_cloud(raw)[1][0], [255,40,40])
        np.testing.assert_array_equal(decode_cloud(edited)[1][0], [0,220,255])
        self.assertEqual(h.mp.image_buffer[0]['rgb_image'][0,0].tolist(), [3, 2, 1])
        folder = next(Path(h.temp.name).iterdir())
        self.assertEqual(json.loads((folder / 'actions_gripper.json').read_text())['action_buffer'], [0,0,1,1])
        for name in ('cmcor_mask.png','cmcor_mask_raw.png','sam3_foreground_mask.png','cmcor_mask_edited.png'):
            self.assertTrue((folder / name).exists(), name)
        self.assertFalse(h.computing)
    def test_missing_sam_mask_keeps_raw_and_does_not_publish_edited(self):
        frames, states = self.data(); h = Harness(); self.addCleanup(h.temp.cleanup)
        h.foreground = TimeoutError('timeout')
        Recorder.compute(h, frames, states)
        self.assertEqual(len(h.clouds['raw']), 1)
        self.assertFalse(h.clouds['edited'])
        self.assertTrue(any('raw retained' in m for m in h.messages))
        self.assertFalse(h.computing)
    def test_unmatched_states_do_not_publish(self):
        frames, states = self.data(); h = Harness(); self.addCleanup(h.temp.cleanup)
        states = [(t+1_000_000_000, a, p) for t, a, p in states]
        Recorder.compute(h, frames, states)
        self.assertFalse(h.clouds['raw'])
        self.assertTrue(any('unmatched' in m for m in h.messages))
        self.assertFalse(h.computing)
    def test_mask_response_must_match_target_frame_stamp_and_dimensions(self):
        from sensor_msgs.msg import Image
        h = Harness(); self.addCleanup(h.temp.cleanup)
        h.mask_condition = threading.Condition(); h.pending_target = (123, 'camera', 2, 2); h.target_mask = None
        msg = Image(); msg.header.stamp.nanosec = 124; msg.header.frame_id = 'camera'
        msg.height = msg.width = 2; msg.encoding = 'mono8'; msg.step = 2; msg.data = bytes([0,255,0,0])
        Recorder.on_target_mask(h, msg)
        self.assertIsNone(h.target_mask)
        msg.header.stamp.nanosec = 123
        Recorder.on_target_mask(h, msg)
        np.testing.assert_array_equal(h.target_mask, h.foreground)


if __name__ == '__main__':
    unittest.main()
