"""Check RViz invocation and installed-config schema without opening a window."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from cable_pointcloud import viewer


class ViewerTests(unittest.TestCase):
    def test_launch_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'rviz' / 'cable_points.rviz'
            config.parent.mkdir()
            config.write_text('Visualization Manager: {}')
            with patch.object(viewer, 'get_package_share_directory', return_value=directory), \
                 patch.object(viewer, 'get_package_prefix', return_value='/opt/ros/humble'), \
                 patch.object(viewer.os, 'execv') as execute, \
                 patch('sys.argv', ['viewer', '--topic', '/custom/cloud', '--fixed-frame', 'camera', '--ros-args', '-r', '__node:=pointcloud_viewer']):
                viewer.main()
            executable, argv = execute.call_args.args
            self.assertEqual(executable, '/opt/ros/humble/lib/rviz2/rviz2')
            self.assertEqual(argv.count('--ros-args'), 1)
            self.assertIn('/cable_camera/cable_points:=/custom/cloud', argv)
            self.assertEqual(argv[argv.index('-f') + 1], 'camera')
            self.assertIn('__node:=pointcloud_viewer', argv)

    def test_picking_config(self):
        config = yaml.safe_load((Path(__file__).parents[1] / 'rviz/cable_points.rviz').read_text())
        manager = config['Visualization Manager']
        cloud = manager['Displays'][0]
        self.assertTrue(cloud['Selectable'])
        self.assertEqual(cloud['Topic']['Reliability Policy'], 'Best Effort')
        self.assertIn('rviz_common/Selection', [p['Class'] for p in config['Panels']])
        tool = next(t for t in manager['Tools'] if t['Class'].endswith('/PublishPoint'))
        self.assertEqual(tool['Topic'], '/clicked_point')


if __name__ == '__main__':
    unittest.main()
