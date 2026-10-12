import unittest
from cable_pointcloud.throughput import Throughput


class ThroughputTests(unittest.TestCase):
    def test_window_rates_and_lifetime_counts(self):
        now = [10.]
        stats = Throughput(('rgb', 'paired'), clock=lambda: now[0])
        for _ in range(30):
            stats.add('rgb')
        stats.add('paired')
        now[0] += 5
        first = stats.snapshot()
        self.assertEqual(first['hz'], {'rgb': 6., 'paired': .2})
        stats.add('rgb')
        now[0] += 2
        second = stats.snapshot()
        self.assertEqual(second['hz'], {'rgb': .5, 'paired': 0.})
        self.assertEqual(second['counts'], {'rgb': 31, 'paired': 1})
        self.assertEqual(second['elapsed_seconds'], 7.)
        self.assertEqual(first['counts']['rgb'], 30)

    def test_empty_window(self):
        stats = Throughput(('rgb',), clock=lambda: 0.)
        self.assertEqual(stats.snapshot()['hz']['rgb'], 0.)
