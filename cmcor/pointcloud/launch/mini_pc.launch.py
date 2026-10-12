from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('publish_fps', default_value='10'),
        DeclareLaunchArgument('crop_left', default_value='100'),
        DeclareLaunchArgument('crop_top', default_value='140'),
        DeclareLaunchArgument('crop_width', default_value='400'),
        DeclareLaunchArgument('crop_height', default_value='300'),
        Node(package='cable_pointcloud', executable='rgbd_capture_publisher',
             arguments=['--width', '640', '--height', '480', '--fps', '30',
                        '--publish-fps', LaunchConfiguration('publish_fps'),
                        '--crop', LaunchConfiguration('crop_left'), LaunchConfiguration('crop_top'),
                        LaunchConfiguration('crop_width'), LaunchConfiguration('crop_height')], output='screen'),
        Node(package='cable_pointcloud', executable='pointcloud_viewer', output='screen')])
