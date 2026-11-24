"""Continuous multi-grasp recording; individual rounds reuse the single recorder."""
import json
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .cmcor_recorder import Recorder


def warp_mask(mask, valid, forward, backward, tolerance):
    """Map source mask to destination with inverse flow and consistency checking."""
    h, w = mask.shape
    if forward.shape != (h, w, 2) or backward.shape != (h, w, 2):
        raise ValueError('optical flow dimensions differ from RGB')
    y, x = np.indices((h, w), dtype=np.float32)
    sx, sy = x + backward[..., 0], y + backward[..., 1]
    finite = np.isfinite(sx) & np.isfinite(sy)
    inside = finite & (sx >= 0) & (sx < w-1) & (sy >= 0) & (sy < h-1)
    sx = np.where(finite, sx, -1).astype(np.float32)
    sy = np.where(finite, sy, -1).astype(np.float32)
    f = cv2.remap(forward.astype(np.float32), sx, sy, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    consistent = np.isfinite(f).all(axis=2) & (np.linalg.norm(f + backward, axis=2) <= tolerance)
    old_valid = cv2.remap(valid.astype(np.uint8), sx, sy, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) != 0
    mapped = cv2.remap(mask.astype(np.uint8), sx, sy, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0) != 0
    new_valid = inside & consistent & old_valid
    return mapped & new_valid, new_valid


class MultiRecorder(Recorder):
    def __init__(self):
        self.session = None
        self.session_lock = threading.RLock()
        self.round_worker = False
        self.active_round = None
        super().__init__()
        self.declare_parameter('session_queue_size', 300)
        self.declare_parameter('max_frame_gap', 0.5)
        self.declare_parameter('flow_consistency_pixels', 1.5)
        self.declare_parameter('min_votes', 1)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.multi_raw_pub = self.create_publisher(PointCloud2, '/cable_camera/cmcor_multi_points_raw', qos)
        self.multi_edited_pub = self.create_publisher(PointCloud2, '/cable_camera/cmcor_multi_points_edited', qos)
        self.create_subscription(String, '/cable_interaction/phase_event', self.on_phase, 100)
        self.create_service(Trigger, '/cable_interaction/start_multigrasp_session', self.start_session)
        self.create_service(Trigger, '/cable_interaction/finish_multigrasp_session', self.finish_session)
        self.create_service(Trigger, '/cable_interaction/abort_multigrasp_session', self.abort_session)

    def enqueue(self, item):
        with self.session_lock:
            s = self.session
            if s is None or not s['accepting'] or s['error']:
                return
            try:
                s['queue'].put_nowait(item)
            except queue.Full:
                s['error'] = 'continuous recording queue overflow; session incomplete'
                self.get_logger().error(s['error'])

    def on_paired_frame(self, frame):
        self.enqueue(('frame', frame))

    def on_state(self, msg):
        super().on_state(msg)
        try:
            data = json.loads(msg.data)
            if int(data['stamp_ns']) > 0:
                self.enqueue(('state', data))
        except (ValueError, TypeError, KeyError):
            pass

    def on_phase(self, msg):
        try:
            data = json.loads(msg.data)
            if int(data['stamp_ns']) > 0:
                self.enqueue(('phase', data))
        except (ValueError, TypeError, KeyError):
            self.get_logger().warning('Invalid phase event')

    def writer(self, s):
        try:
            with (s['folder'] / 'frames.jsonl').open('w') as frames, \
                 (s['folder'] / 'states.jsonl').open('w') as states, \
                 (s['folder'] / 'phases.jsonl').open('w') as phases:
                while True:
                    item = s['queue'].get()
                    try:
                        if item is None:
                            break
                        kind, value = item
                        if kind == 'frame':
                            stamp, fid, rgb, depth, matrix = value
                            i = s['frame_count']
                            rgb_name, depth_name = f'rgb_{i:08d}.png', f'depth_{i:08d}.png'
                            if not cv2.imwrite(str(s['folder'] / rgb_name), rgb) or not cv2.imwrite(str(s['folder'] / depth_name), depth):
                                raise IOError('cannot save RGB-D frame')
                            frames.write(json.dumps(dict(stamp_ns=stamp, frame_id=fid, rgb=rgb_name,
                                                         depth=depth_name, camera_matrix=matrix.tolist())) + '\n')
                            s['frame_count'] += 1
                        elif kind == 'state':
                            states.write(json.dumps(value) + '\n')
                        else:
                            phases.write(json.dumps(value) + '\n')
                    finally:
                        s['queue'].task_done()
        except Exception as exc:
            s['error'] = f'continuous recording failed: {exc}'
            self.get_logger().error(s['error'])

    def start_session(self, req, res):
        with self.lock:
            error = super().readiness_error()
        with self.session_lock:
            if self.session is not None:
                error = 'a multi-grasp session is already active or processing'
            if error:
                res.message = error
                return res
            root = Path(self.get_parameter('output_root').value).expanduser()
            folder = root / ('multi-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
            try:
                folder.mkdir(parents=True)
                (folder / 'rounds').mkdir()
            except OSError as exc:
                res.message = f'cannot create session directory: {exc}'
                return res
            s = dict(folder=folder, queue=queue.Queue(maxsize=max(1, int(self.get_parameter('session_queue_size').value))),
                     accepting=True, error=None, frame_count=0, rounds=[])
            self.session = s
            s['thread'] = threading.Thread(target=self.writer, args=(s,), daemon=True)
            s['thread'].start()
            # Empty volatile messages clear displayed results at the start of a new session.
            from .ros_nodes import cloud_message
            empty = cloud_message({'timestamp_ns': self.last_pair[1], 'frame_id': self.camera[0]}, np.empty((0, 3), np.float32), None)
            self.multi_raw_pub.publish(empty)
            self.multi_edited_pub.publish(empty)
        res.success = True
        res.message = f'continuous session started: {folder}'
        self.report(res.message)
        return res

    def readiness_error(self):
        error = super().readiness_error()
        with self.session_lock:
            if self.session is None or not self.session['accepting']:
                return 'start a multi-grasp session first'
            if self.session['error']:
                return self.session['error']
            if self.round_worker:
                return 'previous round still processing'
        return error

    def start(self, req, res):
        res = super().start(req, res)
        if res.success:
            with self.session_lock:
                s = self.session
                self.active_round = len(s['rounds']) + 1
                s['rounds'].append(dict(number=self.active_round, status='recording', start_stamp_ns=self.last_pair[1]))
        return res

    def finish(self, req, res):
        with self.session_lock:
            self.round_worker = True
        with self.session_lock:
            if self.session and self.active_round and self.last_pair:
                self.session['rounds'][self.active_round-1]['finish_stamp_ns'] = self.last_pair[1]
        res = super().finish(req, res)
        if not res.success:
            with self.session_lock:
                self.round_worker = False
        return res

    def abort(self, req, res):
        res = super().abort(req, res)
        with self.session_lock:
            if self.session and self.active_round:
                self.session['rounds'][self.active_round-1]['status'] = 'aborted'
                self.active_round = None
        return res

    def save_sequence(self, matched):
        source = super().save_sequence(matched)
        with self.session_lock:
            s = self.session
            round_record = s['rounds'][self.active_round-1]
            target = s['folder'] / 'rounds' / f"{self.active_round:04d}"
            shutil.move(str(source), str(target))
            round_record.update(target_stamp_ns=matched[-1][0][0], folder=str(target.relative_to(s['folder'])))
        return target

    def compute(self, frames, states):
        try:
            super().compute(frames, states)
            with self.session_lock:
                record = self.session['rounds'][self.active_round-1]
                folder = self.session['folder'] / record.get('folder', 'missing')
                record['status'] = 'complete' if (folder / 'cmcor_mask_raw.png').exists() else 'failed'
                record['edited'] = (folder / 'cmcor_mask_edited.png').exists()
        finally:
            with self.session_lock:
                self.round_worker = False
                self.active_round = None

    def watchdog(self):
        super().watchdog()
        with self.session_lock:
            failed = self.session is not None and self.session['error']
        if failed:
            with self.lock:
                self.recording = False
                self.frames = []

    def finish_session(self, req, res):
        with self.lock:
            if self.recording or self.computing:
                res.message = 'finish perturbation and wait for round computation first'
                return res
        with self.session_lock:
            s = self.session
            if s is None or not s['accepting']:
                res.message = 'no active session'
                return res
            if self.round_worker:
                res.message = 'wait for round computation first'
                return res
            s['accepting'] = False
        threading.Thread(target=self.finalize, args=(s, False), daemon=True).start()
        res.success = True
        res.message = f'session stopped; alignment started: {s["folder"]}'
        self.report(res.message)
        return res

    def abort_session(self, req, res):
        with self.lock:
            if self.recording or self.computing:
                res.message = 'stop active perturbation and wait for computation first'
                return res
        with self.session_lock:
            s = self.session
            if s is None or not s['accepting'] or self.round_worker:
                res.message = 'no abortable session'
                return res
            s['accepting'] = False
        threading.Thread(target=self.finalize, args=(s, True), daemon=True).start()
        res.success = True
        res.message = 'session aborted; recorded data retained'
        return res

    def finalize(self, s, aborted):
        result = dict(rounds=s['rounds'], status='aborted' if aborted else 'failed')
        try:
            # Writer drains all accepted frames before alignment. A failed writer must not block here.
            if s['thread'].is_alive():
                s['queue'].put(None, timeout=10)
                s['thread'].join(timeout=120)
            if s['thread'].is_alive():
                raise RuntimeError('writer did not finish')
            if s['error']:
                raise RuntimeError(s['error'])
            if not aborted:
                self.fuse(s, result)
                result['status'] = 'complete'
                self.report(f'multi-grasp completed: {s["folder"]}')
        except Exception as exc:
            result['error'] = str(exc)
            self.report(f'multi-grasp failed: {exc}; session={s["folder"]}')
        finally:
            try:
                (s['folder'] / 'session.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            finally:
                with self.session_lock:
                    if self.session is s:
                        self.session = None

    def fuse(self, s, result):
        frames = [json.loads(x) for x in (s['folder'] / 'frames.jsonl').read_text().splitlines()]
        if not frames:
            raise ValueError('session has no RGB-D frames')
        stamps = [f['stamp_ns'] for f in frames]
        max_gap = float(self.get_parameter('max_frame_gap').value)
        if max_gap <= 0 or any(b <= a or (b-a)*1e-9 > max_gap for a, b in zip(stamps, stamps[1:])):
            raise ValueError('non-monotonic RGB-D timestamps or frame gap exceeds max_frame_gap')
        rounds = [r for r in s['rounds'] if r['status'] == 'complete']
        if len(rounds) != len(s['rounds']) or not rounds:
            raise ValueError('no complete rounds or at least one round failed/aborted')
        by_stamp = {stamp: i for i, stamp in enumerate(stamps)}
        entries = {}
        for r in rounds:
            if r['target_stamp_ns'] not in by_stamp:
                raise ValueError('round target frame missing from continuous recording')
            i = by_stamp[r['target_stamp_ns']]
            entries.setdefault(i, []).append(r)
        tolerance = float(self.get_parameter('flow_consistency_pixels').value)
        votes_needed = int(self.get_parameter('min_votes').value)
        if tolerance <= 0 or votes_needed <= 0:
            raise ValueError('invalid consistency threshold or minimum votes')
        active = []
        prev = None
        for i, f in enumerate(frames):
            rgb = cv2.imread(str(s['folder'] / f['rgb']), cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError('missing session RGB')
            if active:
                # The predictor returns source-to-destination flow, in RGB input order.
                forward = self.mp.of_predictor.flow(prev[..., ::-1].copy(), rgb[..., ::-1].copy())
                backward = self.mp.of_predictor.flow(rgb[..., ::-1].copy(), prev[..., ::-1].copy())
                for item in active:
                    item['raw'], item['valid_raw'] = warp_mask(item['raw'], item['valid_raw'], forward, backward, tolerance)
                    if item['edited'] is not None:
                        item['edited'], item['valid_edited'] = warp_mask(item['edited'], item['valid_edited'], forward, backward, tolerance)
            for r in entries.get(i, []):
                folder = s['folder'] / r['folder']
                raw = cv2.imread(str(folder / 'cmcor_mask_raw.png'), cv2.IMREAD_GRAYSCALE)
                edited = cv2.imread(str(folder / 'cmcor_mask_edited.png'), cv2.IMREAD_GRAYSCALE) if r['edited'] else None
                if raw is None or raw.shape != rgb.shape[:2] or (edited is not None and edited.shape != raw.shape):
                    raise ValueError('round mask dimensions do not match session RGB')
                active.append(dict(round=r['number'], raw=raw > 0, edited=edited > 0 if edited is not None else None,
                                   valid_raw=np.ones(raw.shape, bool), valid_edited=np.ones(raw.shape, bool)))
            prev = rgb
        final = frames[-1]
        depth = cv2.imread(str(s['folder'] / final['depth']), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.shape != prev.shape[:2] or depth.dtype != np.uint16:
            raise ValueError('final aligned depth missing or invalid')
        target = (final['stamp_ns'], final['frame_id'], prev, depth, np.asarray(final['camera_matrix']))
        result.update(final_stamp_ns=final['stamp_ns'], frame_count=len(frames), flow_backend=self.mp.of_predictor.backend,
                      max_frame_gap=max_gap, flow_consistency_pixels=tolerance, min_votes=votes_needed)
        for mode in ('raw', 'edited'):
            selected = [a for a in active if a[mode] is not None]
            if not selected or (mode == 'edited' and len(selected) != len(active)):
                result[mode] = 'incomplete: one or more rounds lack an edited mask'
                continue
            votes = sum((a[mode].astype(np.uint16) for a in selected))
            valid = sum((a['valid_'+mode].astype(np.uint16) for a in selected))
            mask = (votes >= votes_needed) & (valid >= votes_needed)
            for a in selected:
                cv2.imwrite(str(s['folder'] / f"round_{a['round']:04d}_{mode}_warped.png"), a[mode].astype(np.uint8)*255)
                cv2.imwrite(str(s['folder'] / f"round_{a['round']:04d}_{mode}_valid.png"), a['valid_'+mode].astype(np.uint8)*255)
            if mode == 'edited':
                foreground = self.request_target_mask(target)
                cv2.imwrite(str(s['folder'] / 'final_sam3_foreground.png'), foreground.astype(np.uint8)*255)
                mask &= foreground
            cv2.imwrite(str(s['folder'] / f'final_{mode}_mask.png'), mask.astype(np.uint8)*255)
            np.save(str(s['folder'] / f'{mode}_votes.npy'), votes)
            np.save(str(s['folder'] / f'{mode}_valid_count.npy'), valid)
            pub, color = ((self.multi_raw_pub, (255, 160, 0)) if mode == 'raw' else
                          (self.multi_edited_pub, (0, 255, 80)))
            result[mode] = dict(points=self.publish_result(target, mask, (pub,), color), rounds=len(selected))


def main():
    rclpy.init()
    node = MultiRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
