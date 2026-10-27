"""Temporary real-cloud relay to isolate BEST_EFFORT fragment loss."""
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

rclpy.init()
node = rclpy.create_node('reliable_cloud_probe')
qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
pub = node.create_publisher(PointCloud2, '/cable_camera/cable_points_reliable_test', qos)
count = [0]
def receive(message):
    pub.publish(message)
    count[0] += 1
sub = node.create_subscription(PointCloud2, '/cable_pointcloud/internal', receive,
                               qos_profile_sensor_data)
started = time.monotonic()
last = started
print('Probe active for 10 minutes; output=/cable_camera/cable_points_reliable_test RELIABLE', flush=True)
try:
    while rclpy.ok() and time.monotonic() - started < 600:
        rclpy.spin_once(node, timeout_sec=.1)
        now = time.monotonic()
        if now - last >= 5:
            print(f'relayed_total={count[0]} subscribers={pub.get_subscription_count()}', flush=True)
            last = now
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
