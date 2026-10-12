from setuptools import setup
from glob import glob

setup(name='cable_pointcloud', version='0.1.0',
      packages=['cable_pointcloud'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/cable_pointcloud']),
                  ('share/cable_pointcloud', ['package.xml']),
                  ('share/cable_pointcloud/launch', glob('launch/*.launch.py'))],
      install_requires=['setuptools'], zip_safe=False,
      maintainer='flexcycle', maintainer_email='maintainer@example.com',
      description='ROS 2 cable point-cloud pipeline', license='MIT',
      entry_points={'console_scripts': [
          'rgbd_capture_publisher = cable_pointcloud.rgbd_capture_publisher:main',
          'rgbd_receiver = cable_pointcloud.rgbd_receiver:main',
          'sam3_pointcloud = cable_pointcloud.ros_nodes:inference_main',
          'pointcloud_publisher = cable_pointcloud.ros_nodes:publisher_main',
          'pointcloud_viewer = cable_pointcloud.ros_nodes:viewer_main']})
