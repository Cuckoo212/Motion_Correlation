"""Start the configured RViz2 viewer, replacing this process for clean shutdown."""
import argparse
import os
import sys
from pathlib import Path
from ament_index_python.packages import get_package_prefix, get_package_share_directory


def main():
    parser = argparse.ArgumentParser(description='RViz2 cable cloud viewer with point selection')
    parser.add_argument('--topic', default='/cable_camera/cable_points')
    parser.add_argument('--fixed-frame', default='cable_camera_color_optical_frame')
    parser.add_argument('--config', type=Path)
    args, ros_args = parser.parse_known_args()
    config = args.config or Path(get_package_share_directory('cable_pointcloud')) / 'rviz' / 'cable_points.rviz'
    if not config.is_file():
        parser.error(f'RViz config not found: {config}')
    try:
        executable = Path(get_package_prefix('rviz2')) / 'lib' / 'rviz2' / 'rviz2'
    except LookupError:
        parser.error('RViz2 is missing: install ros-humble-rviz2 on the Mini PC')
    command = [str(executable), '-d', str(config), '-f', args.fixed_frame, *ros_args]
    if '--ros-args' not in ros_args:
        command.append('--ros-args')
    command.extend(['-r', '/cable_camera/cable_points:=' + args.topic])
    os.execv(str(executable), command)


if __name__ == '__main__':
    main()
