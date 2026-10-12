"""Bounded in-memory RGB-D pairing and latest-frame cache."""
from __future__ import annotations
import threading
import time
import numpy as np

MAX_PIXELS = 4096 * 4096


def decode_image(message, encoding):
    """Decode sensor_msgs/Image with padded rows and explicit byte order."""
    if message.encoding != encoding:
        raise ValueError(f'Expected {encoding}, got {message.encoding}')
    channels, itemsize = (3, 1) if encoding == 'bgr8' else (1, 2)
    row_size = message.width * channels * itemsize
    if min(message.width, message.height) <= 0 or message.width * message.height > MAX_PIXELS:
        raise ValueError('Invalid image dimensions')
    if message.step < row_size or len(message.data) != message.step * message.height:
        raise ValueError('Invalid image stride or buffer size')
    rows = np.frombuffer(bytes(message.data), np.uint8).reshape(message.height, message.step)
    packed = rows[:, :row_size].copy()
    if encoding == 'bgr8':
        return packed.reshape(message.height, message.width, 3)
    dtype = '>u2' if message.is_bigendian else '<u2'
    return np.frombuffer(packed.tobytes(), dtype).reshape(message.height, message.width).astype(np.uint16)


class FramePairer:
    """Exact timestamp pairing; unmatched messages never grow without bound."""
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.pending = {}
        self.last_stamp = -1

    def add(self, kind, stamp, frame_id, array):
        pair = self.pending.setdefault(stamp, {})
        pair[kind] = (frame_id, array)
        result = None
        if 'rgb' in pair and 'depth' in pair:
            del self.pending[stamp]
            rgb_id, rgb = pair['rgb']
            depth_id, depth = pair['depth']
            if rgb_id != depth_id or rgb.shape[:2] != depth.shape:
                raise ValueError('RGB-D frame IDs or dimensions differ')
            if stamp > self.last_stamp:
                self.last_stamp = stamp
                result = (rgb_id, rgb, depth)
        while len(self.pending) > self.capacity:
            del self.pending[min(self.pending)]
        return result


class LatestFrame:
    def __init__(self):
        self.condition = threading.Condition()
        self.version = 0
        self.frame = None

    def put(self, frame):
        with self.condition:
            self.frame = frame
            self.version += 1
            self.condition.notify_all()

    def get_after(self, version, stop, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self.version <= version and not stop.is_set():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("No fresh paired RGB-D frame")
                self.condition.wait(.2 if remaining is None else min(.2, remaining))
            if stop.is_set():
                return None
            return self.version, self.frame
