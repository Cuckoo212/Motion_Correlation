import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    root = LaunchConfiguration('project_root')
    inference = Node(package='cable_pointcloud', executable='sam3_pointcloud',
                     prefix=[os.environ.get('SAM3_PYTHON', '/opt/sam3-env/bin/python')],
                     arguments=['--topic-prefix', LaunchConfiguration('topic_prefix'), '--no-view', '--receive-timeout', '60',
                                '--sam3-project', PathJoinSubstitution([root, 'pointcloud', 'sam3_project']),
                                '--dataset-root', PathJoinSubstitution([root, 'datasets', 'realtime_cable_pc']),
                                '--output-root', PathJoinSubstitution([root, 'output', 'realtime_cable_pc'])],
                     output='screen')
    publisher = Node(package='cable_pointcloud', executable='pointcloud_publisher', output='screen')
    return LaunchDescription([
        DeclareLaunchArgument('project_root', default_value='/workspace/cmcor'),
        DeclareLaunchArgument('topic_prefix', default_value='/cable_camera'),
        publisher, inference,
        *[RegisterEventHandler(OnProcessExit(target_action=process,
             on_exit=[EmitEvent(event=Shutdown(reason='GPU pipeline node exited'))]))
          for process in (inference, publisher)]])
