#!/usr/bin/env python3
"""Repeat SAM3 on recorded RGB images without ROS/DDS for crash isolation."""
import argparse
from pathlib import Path
import time
import cv2
from cable_pointcloud.sam3_segmenter import Sam3CableSegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path, help='Directory containing rgb_*.png')
    parser.add_argument('--frames', type=int, default=5000)
    parser.add_argument('--sam3-project', type=Path,
                        default=Path('/workspace/cmcor/pointcloud/sam3_project'))
    args = parser.parse_args()
    paths = sorted(args.images.glob('rgb_*.png'))
    if not paths or args.frames <= 0:
        parser.error('Need RGB images and a positive frame count')
    segmenter = Sam3CableSegmenter(args.sam3_project,
                                  args.sam3_project / 'checkpoints/sam3.pt')
    for index in range(args.frames):
        path = paths[index % len(paths)]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Cannot read {path}')
        started = time.monotonic()
        mask, score = segmenter.segment(image)
        print(f'frame={index:08d} image={path.name} '
              f'SAM={(time.monotonic()-started)*1000:.1f}ms score={score:.4f}', flush=True)


if __name__ == '__main__':
    main()
