"""Arguments for the ROS SAM3 calculation node."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/workspace/cmcor')
    parser.add_argument('--sam3-project', type=Path, default=root/'pointcloud/sam3_project')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dataset-root', type=Path, default=root/'datasets/realtime_cable_pc')
    parser.add_argument('--output-root', type=Path, default=root/'output/realtime_cable_pc')
    parser.add_argument('--sequence-name')
    parser.add_argument('--socket-path', type=Path, default=Path('/run/cable-rgbd/cable_rgbd.sock'))
    parser.add_argument('--receive-timeout', type=float, default=60.)
    parser.add_argument('--prompt', default='cable')
    parser.add_argument('--confidence-threshold', type=float, default=.5)
    parser.add_argument('--min-area-ratio', type=float, default=.0001)
    parser.add_argument('--max-area-ratio', type=float, default=.5)
    parser.add_argument('--selection', choices=['union', 'best'], default='union')
    parser.add_argument('--min-depth-mm', type=int, default=150)
    parser.add_argument('--max-depth-mm', type=int, default=2000)
    parser.add_argument('--voxel-size-mm', type=float, default=2.)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--no-view', action='store_true')
    parser.add_argument('--view-width', type=int, default=1280)
    parser.add_argument('--view-height', type=int, default=800)
    parser.add_argument('--point-size', type=float, default=8.)
    parser.add_argument('--view-rgb', action='store_true')
    return parser.parse_args()
