from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('publish_fps', default_value='10'),
        Node(package='cable_pointcloud', executable='rgbd_capture_publisher',
             arguments=['--width', '640', '--height', '480', '--fps', '30',
                        '--publish-fps', LaunchConfiguration('publish_fps')], output='screen'),
        Node(package='cable_pointcloud', executable='pointcloud_viewer', output='screen')])
