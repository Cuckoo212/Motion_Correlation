"""Monotonic window counters: rates never depend on cross-machine clock sync."""
import time


class Throughput:
    def __init__(self, names, clock=time.monotonic):
        self.clock = clock
        self.started = self.previous_time = clock()
        self.counts = dict.fromkeys(names, 0)
        self.previous = self.counts.copy()

    def add(self, name):
        self.counts[name] += 1

    def snapshot(self):
        now = self.clock()
        seconds = now - self.previous_time
        value = dict(window_seconds=seconds, elapsed_seconds=now - self.started,
                     counts=self.counts.copy(),
                     hz={name: (count - self.previous[name]) / seconds if seconds > 0 else 0.
                         for name, count in self.counts.items()})
        self.previous_time, self.previous = now, self.counts.copy()
        return value
