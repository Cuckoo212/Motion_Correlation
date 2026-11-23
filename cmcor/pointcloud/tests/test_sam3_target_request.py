import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from cable_pointcloud.rgbd_transport import LatestFrame
from cable_pointcloud import inference


class TargetRequestTest(unittest.TestCase):
    def test_target_request_uses_requested_image_and_preserves_header(self):
        with tempfile.TemporaryDirectory() as directory:
            source = LatestFrame(); stop = threading.Event(); requests = queue.Queue(maxsize=1)
            live = np.full((2,3,3), 50, np.uint8)
            target = np.full((2,3,3), 100, np.uint8)
            header = SimpleNamespace(stamp=123, frame_id='camera')
            requests.put((header, target))
            metadata = dict(timestamp_ns=123, received_timestamp_ns=456, received_pair_index=0,
                frame_id='camera', camera_info={'camera_matrix': [[100,0,1],[0,100,1],[0,0,1]]})
            source.put((metadata, live, np.full((2,3), 1000, np.uint16)))
            outputs = []
            argv = ['pc', '--dataset-root', directory, '--output-root', directory,
                    '--no-view', '--max-frames', '1']
            def segment(img):
                return np.full((2,3), int(img[0,0,0]), np.uint8), .9
            with patch('sys.argv', argv), patch.object(inference, 'Sam3CableSegmenter') as model:
                model.return_value.segment.side_effect = segment
                inference.main(source, stop_event=stop, mask_requests=requests,
                               on_requested_mask=lambda h, m: outputs.append((h,m)))
            self.assertIs(outputs[0][0], header)
            np.testing.assert_array_equal(outputs[0][1], np.full((2,3), 100, np.uint8))
            self.assertEqual(model.return_value.segment.call_count, 2)
            np.testing.assert_array_equal(model.return_value.segment.call_args_list[1].args[0], live)


if __name__ == '__main__':
    unittest.main()
