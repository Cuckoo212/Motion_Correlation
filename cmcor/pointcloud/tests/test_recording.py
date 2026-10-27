import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from cable_pointcloud import inference
from cable_pointcloud.inference_args import parse_args


class RecordingTests(unittest.TestCase):
    def test_record_flag_defaults_and_values(self):
        for flags, expected in (([], False), (['--record'], True),
                                (['--record', 'true'], True), (['--record', 'false'], False)):
            with patch('sys.argv', ['node'] + flags):
                self.assertEqual(parse_args().record, expected)

    def test_default_publishes_without_any_recording_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = dict(timestamp_ns=1, received_timestamp_ns=2, received_pair_index=0,
                            frame_id='camera', camera_info=dict(camera_matrix=np.eye(3).tolist()))
            source = Mock()
            source.get_after.side_effect = [
                (i+1, (dict(metadata, received_pair_index=i, received_timestamp_ns=i*10**9),
                       np.zeros((2, 3, 3), np.uint8), np.full((2, 3), 1000, np.uint16)))
                for i in range(3)]
            argv = ['node', '--dataset-root', str(root/'data'), '--output-root', str(root/'output'),
                    '--no-view', '--max-frames', '3']
            output = io.StringIO()
            published = Mock()
            with patch('sys.argv', argv), patch.object(inference, 'Sam3CableSegmenter') as segmenter, \
                 patch.object(inference.cv2, 'imwrite') as image_write, \
                 patch.object(inference, 'save_ply') as ply_write, \
                 patch.object(inference, 'atomic_json') as json_write, contextlib.redirect_stdout(output):
                segmenter.return_value.segment.return_value = (np.full((2, 3), 255, np.uint8), .9)
                inference.main(source, on_cloud=published)
                image_write.assert_not_called()
                ply_write.assert_not_called()
                json_write.assert_not_called()
            self.assertEqual(published.call_count, 3)
            self.assertEqual(list(root.iterdir()), [])
            self.assertIn('frame=00000002', output.getvalue())
            self.assertIn('[rates final]', output.getvalue())

    def test_cumulative_rates_do_not_require_frame_history(self):
        stats = inference.RunStats()
        for i in range(100):
            stats.add(dict(received_timestamp_ns=i*10**9, received_pair_index=i*2,
                           camera_info={'fps': 30, 'publish_fps': 30}),
                      dict(completed_monotonic_s=i*.5, sam_inference_ms=100., result_age_ms=200.))
        summary = stats.summary()
        self.assertEqual(summary['receiver_hz'], 2.)
        self.assertEqual(summary['cloud_hz'], 2.)
        self.assertEqual(summary['mean_sam_inference_ms'], 100.)
        self.assertEqual(summary['skipped_pairs'], 99)
