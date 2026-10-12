"""ROS inference, PointCloud2 relay and Mini PC display entry points."""
import sys
import threading
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


def cloud_message(metadata, points, colors):
    message = PointCloud2()
    message.header.frame_id = metadata['frame_id']
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(metadata['timestamp_ns'], 10**9)
    message.height, message.width = 1, len(points)
    message.fields = [PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(('x', 'y', 'z'))]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.width * message.point_step
    packed = np.zeros((len(points), 4), dtype='<f4')
    packed[:, :3] = points
    if colors is not None:
        rgb = np.asarray(colors, dtype=np.uint32)
        packed.view('<u4')[:, 3] = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    message.fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1))
    message.data = packed.tobytes()
    message.is_dense = bool(np.isfinite(points).all())
    return message


def decode_cloud(message):
    if message.point_step != 16 or message.height != 1 or message.is_bigendian:
        raise ValueError('Expected little-endian XYZ/RGB cloud from cable_pointcloud')
    if len(message.data) != message.width * 16:
        raise ValueError('Invalid PointCloud2 data length')
    fields = {f.name: (f.offset, f.datatype, f.count) for f in message.fields}
    if any(fields.get(n) != (i*4, PointField.FLOAT32, 1) for i, n in enumerate(('x', 'y', 'z'))):
        raise ValueError('Invalid XYZ fields')
    if fields.get('rgb') != (12, PointField.UINT32, 1):
        raise ValueError('Invalid RGB field')
    packed = np.frombuffer(bytes(message.data), dtype='<f4').reshape(-1, 4)
    rgb = packed.view('<u4')[:, 3]
    colors = np.column_stack(((rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255)).astype(np.uint8)
    valid = np.isfinite(packed[:, :3]).all(axis=1)
    return packed[valid, :3].copy(), colors[valid]


def inference_main():
    # Keep existing CLI flags and recording behavior; remove ROS-specific arguments.
    argv = rclpy.utilities.remove_ros_args(sys.argv)
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('sam3_pointcloud')
    pub = node.create_publisher(PointCloud2, '/cable_pointcloud/internal', qos_profile_sensor_data)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    original = sys.argv
    try:
        from .inference import main
        sys.argv = argv
        node.get_logger().info('SAM3 ROS node started; publishing computed clouds')
        main(on_cloud=lambda metadata, points, colors: pub.publish(cloud_message(metadata, points, colors)))
    finally:
        sys.argv = original
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(2)
        node.destroy_node()


def publisher_main():
    rclpy.init()
    node = rclpy.create_node('pointcloud_publisher')
    node.declare_parameter('input_topic', '/cable_pointcloud/internal')
    node.declare_parameter('output_topic', '/cable_camera/cable_points')
    pub = node.create_publisher(PointCloud2, node.get_parameter('output_topic').value, qos_profile_sensor_data)
    node.create_subscription(PointCloud2, node.get_parameter('input_topic').value, pub.publish, qos_profile_sensor_data)
    node.get_logger().info('Publishing cable PointCloud2 to Mini PC')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def viewer_main():
    from .runtime import MatplotlibViewer
    rclpy.init()
    node = rclpy.create_node('pointcloud_viewer')
    node.declare_parameter('topic', '/cable_camera/cable_points')
    node.declare_parameter('view_rgb', False)
    viewer = MatplotlibViewer(True, use_rgb=node.get_parameter('view_rgb').value)
    latest = []
    def receive(message):
        try:
            latest[:] = [decode_cloud(message)]
        except ValueError as exc:
            node.get_logger().warning(str(exc))
    node.create_subscription(PointCloud2, node.get_parameter('topic').value, receive, qos_profile_sensor_data)
    try:
        while rclpy.ok() and not viewer.closed:
            rclpy.spin_once(node, timeout_sec=.01)
            if latest and not viewer.update(*latest.pop()):
                break
            viewer.plt.pause(.01)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
