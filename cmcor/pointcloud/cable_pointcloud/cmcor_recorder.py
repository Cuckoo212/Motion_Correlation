"""Collect synchronized RGB-D and perturbation state, then publish a CMCor cloud."""
import json
import threading
import time
import uuid
from pathlib import Path
import cv2
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from .rgbd_transport import FramePairer, decode_image
from .geometry import depth_mask_to_points
from .ros_nodes import cloud_message


def ns(stamp):
    return stamp.sec * 1000000000 + stamp.nanosec


class Recorder(Node):
    def __init__(self):
        super().__init__('cmcor_recorder')
        self.declare_parameter('topic_prefix', '/cable_camera')
        self.declare_parameter('state_topic', '/cable_interaction/state')
        self.declare_parameter('max_state_age', 0.10)
        self.declare_parameter('min_frames_per_action', 8)
        self.declare_parameter('mask_threshold', 0.86)
        self.declare_parameter('max_frames', 1500)
        self.declare_parameter('gpu', True)
        self.declare_parameter('output_root', '/workspace/cmcor/output/cmcor_sequences')
        prefix = self.get_parameter('topic_prefix').value.rstrip('/')
        sensor = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        info = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        from cmcor.motion_perception import MotionPerception
        self.mp = MotionPerception(gpu=bool(self.get_parameter('gpu').value))
        self.camera = None
        self.metadata_ok = False
        self.pairer = FramePairer(capacity=30)
        self.last_pair = None
        self.last_state_received = None
        self.states = deque(maxlen=3000)
        self.frames = []
        self.recording = False
        self.computing = False
        self.lock = threading.Lock()
        self.pub = self.create_publisher(__import__('sensor_msgs.msg', fromlist=['PointCloud2']).PointCloud2, '/cable_camera/cmcor_points', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.status = self.create_publisher(String, '/cable_interaction/cmcor_status', 10)
        self.heartbeat = self.create_publisher(Bool, '/cable_interaction/recording_active', 10)
        self.create_subscription(CameraInfo, prefix+'/color/camera_info', self.on_info, info)
        self.create_subscription(String, prefix+'/metadata', self.on_metadata, info)
        self.create_subscription(Image, prefix+'/color/image_raw', lambda m: self.on_image('rgb', m), sensor)
        self.create_subscription(Image, prefix+'/aligned_depth/image_raw', lambda m: self.on_image('depth', m), sensor)
        self.create_subscription(String, self.get_parameter('state_topic').value, self.on_state, 100)
        self.create_service(Trigger, '/cable_interaction/ready_recording', self.ready)
        self.create_service(Trigger, '/cable_interaction/start_recording', self.start)
        self.create_service(Trigger, '/cable_interaction/finish_recording', self.finish)
        self.create_service(Trigger, '/cable_interaction/abort_recording', self.abort)
        self.create_timer(.2, self.send_heartbeat)
        self.create_timer(.5, self.watchdog)

    def report(self, text):
        self.get_logger().info(text)
        self.status.publish(String(data=text))

    def on_metadata(self, msg):
        try:
            data = json.loads(msg.data)
            self.metadata_ok = data.get('color_aligned_depth') is True and data.get('depth_unit') == 'millimetre in saved PNG'
        except (ValueError, TypeError):
            self.metadata_ok = False

    def send_heartbeat(self):
        with self.lock:
            self.heartbeat.publish(Bool(data=self.recording))

    def watchdog(self):
        with self.lock:
            if self.recording and (self.last_state_received is None or
                    time.monotonic() - self.last_state_received > 1.0 or
                    self.last_pair is None or time.monotonic() - self.last_pair[0] > 1.0):
                self.recording = False
                self.frames = []
                self.report('aborted: robot state or RGB-D stream timed out')

    def on_info(self, msg):
        matrix = np.asarray(msg.k).reshape(3, 3)
        if not msg.header.frame_id or min(msg.width, msg.height) <= 0 or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            self.camera = None
            self.get_logger().warning('Invalid camera calibration')
            return
        self.camera = (msg.header.frame_id, msg.width, msg.height, matrix)

    def on_state(self, msg):
        try:
            s = json.loads(msg.data)
            t, action, p = int(s['stamp_ns']), int(s['action']), np.asarray(s['point_camera'], dtype=float)
            if t <= 0 or action not in (-1, 0, 1) or p.shape != (3,) or not np.isfinite(p).all() or p[2] <= 0:
                raise ValueError('invalid state')
            with self.lock:
                if self.states and t <= self.states[-1][0]:
                    return
                self.last_state_received = time.monotonic()
                # Bound a 1 kHz robot state stream without losing action transitions.
                if not self.states or action != self.states[-1][1] or t-self.states[-1][0] >= 20_000_000:
                    self.states.append((t, action, p))
        except (ValueError, TypeError, KeyError) as e:
            self.get_logger().warning(f'Invalid interaction state: {e}')

    def on_image(self, kind, msg):
        try:
            t = ns(msg.header.stamp)
            pair = self.pairer.add(kind, t, msg.header.frame_id, decode_image(msg, 'bgr8' if kind == 'rgb' else '16UC1'))
            if pair is None:
                return
            fid, rgb, depth = pair
            if not self.metadata_ok or self.camera is None or self.camera[:3] != (fid, rgb.shape[1], rgb.shape[0]):
                return
            with self.lock:
                self.last_pair = (time.monotonic(), t)
                if self.recording:
                    if len(self.frames) >= int(self.get_parameter('max_frames').value):
                        self.recording = False
                        self.frames = []
                        self.report('aborted: frame limit exceeded')
                    else:
                        self.frames.append((t, fid, rgb, depth, self.camera[3].copy()))
        except (ValueError, TypeError) as e:
            self.get_logger().warning(f'Invalid RGB-D: {e}')

    def readiness_error(self):
        if self.recording or self.computing:
            return 'recorder busy'
        if self.last_pair is None or time.monotonic()-self.last_pair[0] > 1.0:
            return 'no fresh aligned RGB-D'
        if not self.states or abs(self.last_pair[1]-self.states[-1][0]) > 500_000_000:
            return 'no recent time-aligned robot state; check clock synchronization'
        return None

    def ready(self, req, res):
        with self.lock:
            error = self.readiness_error()
            res.success = error is None
            res.message = error or 'ready'
        return res

    def start(self, req, res):
        with self.lock:
            error = self.readiness_error()
            if error:
                res.message = error
            else:
                self.frames = []
                self.recording = True
                # Clear any previous latched result before starting a new sequence.
                self.pub.publish(cloud_message({'timestamp_ns': self.last_pair[1], 'frame_id': self.camera[0]}, np.empty((0, 3), np.float32), None))
                res.success = True
                res.message = 'recording ready'
        return res

    def abort(self, req, res):
        with self.lock:
            self.recording = False
            self.frames = []
        res.success = True
        res.message = 'recording aborted'
        return res

    def finish(self, req, res):
        with self.lock:
            if not self.recording:
                res.message = 'no active recording'
                return res
            self.recording = False
            frames, states = self.frames, list(self.states)
            self.frames = []
            self.computing = True
        threading.Thread(target=self.compute, args=(frames, states), daemon=True).start()
        res.success = True
        res.message = 'recording received; computation started'
        return res

    def save_sequence(self, matched):
        root = Path(self.get_parameter('output_root').value).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        folder = root / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
        folder.mkdir()
        actions, positions, stamps = [], [], []
        for index, ((stamp, _, rgb, depth, _), (_, action, point)) in enumerate(matched):
            if not cv2.imwrite(str(folder / f'rgb_{index:08d}.png'), rgb):
                raise IOError('Failed to save RGB frame')
            if not cv2.imwrite(str(folder / f'depth_{index:08d}.png'), depth):
                raise IOError('Failed to save depth frame')
            actions.append(action)
            positions.append(point.tolist())
            stamps.append(stamp)
        metadata = {'action_buffer': actions, 'ee_point_buffer': positions,
                    'camera_matrix': matched[-1][0][4].tolist(),
                    'gripper_depth': float(np.median([s[2][2] for _, s in matched if s[1] >= 0])),
                    'poor_arm_masks': True, 'point_frame': 'camera_optical',
                    'timestamp_ns': stamps, 'crop_origin': None, 'crop_size': None, 'scale_factor': None}
        (folder / 'actions_gripper.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return folder

    def compute(self, frames, states):
        try:
            max_delta = float(self.get_parameter('max_state_age').value) * 1e9
            matched = []
            for f in frames:
                candidates = [s for s in states if abs(s[0]-f[0]) <= max_delta]
                if candidates:
                    s = min(candidates, key=lambda x: abs(x[0]-f[0]))
                    matched.append((f, s))
            minimum = int(self.get_parameter('min_frames_per_action').value)
            counts = [sum(s[1] == a for _, s in matched) for a in (0, 1)]
            if len(matched) < .8 * len(frames):
                raise ValueError(f'too many unmatched frames: received={len(frames)} matched={len(matched)}')
            if min(counts) < minimum:
                raise ValueError(f'insufficient synchronized frames per action: {counts}; received={len(frames)} matched={len(matched)}')
            sequence = self.save_sequence(matched)
            mp = self.mp
            mp.clear_buffer()
            for (_, _, rgb, depth, _), (_, action, point) in matched:
                mp.add_sample(rgb[..., ::-1].copy(), None, point.tolist(), action, depth_image=depth)
            target, _ = matched[-1]
            matrix = target[4]
            depth_m = float(np.median([s[2][2] for _, s in matched if s[1] >= 0]))
            mp.compute_correlation_images(depth_m, float(matrix[0, 0]))
            if mp.corr_imgs is None or not mp.action2index:
                raise ValueError('CMCor returned no correlation images')
            threshold = float(self.get_parameter('mask_threshold').value)
            masks = [np.nan_to_num(mp.corr_imgs[i], nan=0., posinf=0., neginf=0.) > threshold for i in mp.action2index.values()]
            mask = np.logical_or.reduce(masks).astype(np.uint8)
            if not cv2.imwrite(str(sequence / 'cmcor_mask.png'), mask * 255):
                raise IOError('Failed to save CMCor mask')
            _, fid, rgb, depth, _ = target
            points, colors = depth_mask_to_points(depth, mask, matrix, rgb)
            if not len(points):
                raise ValueError('CMCor mask has no valid depth points')
            self.pub.publish(cloud_message({'timestamp_ns': target[0], 'frame_id': fid}, points, colors))
            self.report(f'completed: frames={len(frames)} matched={len(matched)} action_frames={counts} points={len(points)} sequence={sequence}')
        except Exception as e:
            self.get_logger().error(f'CMCor failed: {e}')
            self.report(f'failed: {e}')
        finally:
            with self.lock:
                self.computing = False


def main():
    rclpy.init()
    node = Recorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
