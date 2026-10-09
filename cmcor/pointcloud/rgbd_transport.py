"""Bounded RGB-D pairing and a local, request-driven Unix socket protocol.

Wire format: uint32 network-endian JSON length, JSON, then BGR8 and LE uint16.
Only trusted local processes should connect. No pickle or remote socket listener.
"""
from __future__ import annotations
import json
import socket
import struct
import threading
import numpy as np

MAX_HEADER = 65536
MAX_PIXELS = 4096 * 4096


def recv_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = connection.recv(size - len(chunks))
        if not part:
            raise EOFError('RGB-D receiver disconnected')
        chunks.extend(part)
    return bytes(chunks)


def send_frame(connection, metadata, rgb, depth):
    height, width = depth.shape
    header = dict(metadata, width=width, height=height, protocol_version=1)
    encoded = json.dumps(header).encode('utf-8')
    if len(encoded) > MAX_HEADER:
        raise ValueError('RGB-D metadata too large')
    connection.sendall(struct.pack('!I', len(encoded)) + encoded)
    connection.sendall(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
    connection.sendall(np.ascontiguousarray(depth, dtype='<u2').tobytes())


def receive_frame(connection):
    length = struct.unpack('!I', recv_exact(connection, 4))[0]
    if not 0 < length <= MAX_HEADER:
        raise ValueError('Invalid RGB-D header size')
    metadata = json.loads(recv_exact(connection, length))
    width, height = int(metadata['width']), int(metadata['height'])
    if metadata.get('protocol_version') != 1 or min(width, height) <= 0 or width * height > MAX_PIXELS:
        raise ValueError('Invalid RGB-D dimensions or version')
    rgb = np.frombuffer(recv_exact(connection, width * height * 3), np.uint8).reshape(height, width, 3).copy()
    depth = np.frombuffer(recv_exact(connection, width * height * 2), '<u2').reshape(height, width).copy()
    return metadata, rgb, depth


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

    def get_after(self, version, stop):
        with self.condition:
            while self.version <= version and not stop.is_set():
                self.condition.wait(0.2)
            if stop.is_set():
                return None
            return self.version, self.frame
