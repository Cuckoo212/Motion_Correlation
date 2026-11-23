import threading
import tempfile
from pathlib import Path
import json
import unittest
from types import SimpleNamespace
import numpy as np
from cable_pointcloud.cmcor_recorder import Recorder


class FakeMotion:
    def __init__(self):
        self.clear_buffer()
    def clear_buffer(self):
        self.samples = []
    def add_sample(self, rgb, mask, point, action, depth_image):
        self.samples.append((rgb, point, action))
    def compute_correlation_images(self, depth, focal):
        self.args = (depth, focal)
        self.action2index = {0: 0, 1: 1}
        self.corr_imgs = [np.array([[.95, 0], [0, 0]]), np.array([[0, 0], [0, .96]])]


class Harness:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.mp = FakeMotion()
        self.msg = None
        self.messages = []
        self.computing = True
        self.lock = threading.Lock()
        self.pub = SimpleNamespace(publish=lambda msg: setattr(self, 'msg', msg))
    def get_parameter(self, key):
        return SimpleNamespace(value={'max_state_age': .1, 'min_frames_per_action': 2, 'mask_threshold': .86, 'output_root': self.temp.name}[key])
    def save_sequence(self, matched):
        return Recorder.save_sequence(self, matched)
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
    def test_synchronized_masks_project_to_target_frame(self):
        frames, states = self.data()
        h = Harness()
        Recorder.compute(h, frames, states)
        self.assertIsNotNone(h.msg)
        self.assertEqual(h.msg.width, 2)
        self.assertEqual(h.msg.header.frame_id, 'camera')
        self.assertEqual(h.msg.header.stamp.nanosec, 150_000_000)
        self.assertEqual(h.mp.args, (1., 100.))
        self.assertEqual(h.mp.samples[0][0][0,0].tolist(), [3, 2, 1])
        self.assertFalse(h.computing)
        folders = list(Path(h.temp.name).iterdir())
        self.assertEqual(len(folders), 1)
        metadata = json.loads((folders[0] / 'actions_gripper.json').read_text())
        self.assertEqual(metadata['action_buffer'], [0, 0, 1, 1])
        self.assertTrue((folders[0] / 'cmcor_mask.png').exists())
    def test_unmatched_states_do_not_publish(self):
        frames, states = self.data()
        states = [(t+1_000_000_000, a, p) for t, a, p in states]
        h = Harness()
        Recorder.compute(h, frames, states)
        self.assertIsNone(h.msg)
        self.assertTrue(any('unmatched' in m for m in h.messages))
        self.assertFalse(h.computing)


if __name__ == '__main__':
    unittest.main()
