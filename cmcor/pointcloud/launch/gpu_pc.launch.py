import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    root = LaunchConfiguration('project_root')
    inference = Node(package='cable_pointcloud', executable='sam3_pointcloud',
                     prefix=[os.environ.get('SAM3_PYTHON', '/opt/sam3-env/bin/python')],
                     arguments=['--record', LaunchConfiguration('record'),
                                '--topic-prefix', LaunchConfiguration('topic_prefix'), '--no-view', '--receive-timeout', '60',
                                '--sam3-project', PathJoinSubstitution([root, 'pointcloud', 'sam3_project']),
                                '--dataset-root', PathJoinSubstitution([root, 'datasets', 'realtime_cable_pc']),
                                '--output-root', PathJoinSubstitution([root, 'output', 'realtime_cable_pc'])],
                     output='screen')
    def recorder(mode, executable):
        return Node(package='cable_pointcloud', executable=executable,
                 condition=IfCondition(PythonExpression(["'", LaunchConfiguration('cmcor_mode'), "' == '", mode, "'"])),
                 prefix=[os.environ.get('CMCOR_PYTHON', os.environ.get('SAM3_PYTHON', '/opt/sam3-env/bin/python'))],
                 additional_env={'PYTHONPATH': [root, ':', EnvironmentVariable('PYTHONPATH', default_value='')]},
                 parameters=[{'topic_prefix': LaunchConfiguration('topic_prefix'),
                              'output_root': PathJoinSubstitution([root, 'output', 'cmcor_sequences'])}] +
                            ([{'max_frame_gap': LaunchConfiguration('max_frame_gap'),
                               'flow_consistency_pixels': LaunchConfiguration('flow_consistency_pixels'),
                               'min_votes': LaunchConfiguration('min_votes'),
                               'session_queue_size': LaunchConfiguration('session_queue_size')}] if mode == 'multi' else []), output='screen')
    publisher = Node(package='cable_pointcloud', executable='pointcloud_publisher', output='screen',
                     parameters=[{'output_reliability': LaunchConfiguration('cloud_reliability')}])
    return LaunchDescription([
        DeclareLaunchArgument('cmcor_mode', default_value='single', choices=['single', 'multi']),
        DeclareLaunchArgument('max_frame_gap', default_value='0.5'),
        DeclareLaunchArgument('flow_consistency_pixels', default_value='1.5'),
        DeclareLaunchArgument('min_votes', default_value='1'),
        DeclareLaunchArgument('session_queue_size', default_value='300'),
        DeclareLaunchArgument('project_root', default_value='/workspace/cmcor'),
        DeclareLaunchArgument('topic_prefix', default_value='/cable_camera'),
        DeclareLaunchArgument('record', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('cloud_reliability', default_value='reliable',
                              choices=['reliable', 'best_effort']),
        publisher, inference, recorder('single', 'cmcor_recorder'),
        recorder('multi', 'cmcor_multigrasp_recorder'),
        *[RegisterEventHandler(OnProcessExit(target_action=process,
             on_exit=[EmitEvent(event=Shutdown(reason='GPU pipeline node exited'))]))
          for process in (inference, publisher)]])
