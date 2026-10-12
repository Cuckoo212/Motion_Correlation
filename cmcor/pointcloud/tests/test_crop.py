import unittest
import numpy as np
from cable_pointcloud.rgbd_capture_publisher import crop_geometry
from cable_pointcloud.geometry import depth_mask_to_points


class CropTests(unittest.TestCase):
    def test_projection_keeps_camera_coordinates(self):
        original = [[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]
        roi, cropped = crop_geometry(640, 480, original, [100, 140, 400, 300])
        self.assertEqual(roi, (100, 140, 400, 300))
        self.assertEqual(cropped[0][2], 220.)
        self.assertEqual(cropped[1][2], 100.)
        depth = np.full((480, 640), 1000, np.uint16)
        mask = np.zeros(depth.shape, np.uint8)
        mask[200, 150] = 255
        full, _ = depth_mask_to_points(depth, mask, np.asarray(original))
        partial, _ = depth_mask_to_points(depth[140:440, 100:500], mask[140:440, 100:500], np.asarray(cropped))
        np.testing.assert_allclose(full, partial)
        self.assertEqual(original[0][2], 320.)

    def test_bounds_and_uncropped(self):
        matrix = np.eye(3)
        self.assertEqual(crop_geometry(640, 480, matrix, None)[0], (0, 0, 640, 480))
        for crop in ([100, 140, 600, 300], [-1, 0, 1, 1], [0, 0, 0, 1], [0, 400, 100, 100]):
            with self.assertRaises(ValueError):
                crop_geometry(640, 480, matrix, crop)


if __name__ == '__main__':
    unittest.main()
