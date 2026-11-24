"""Deterministic checks for temporal mask propagation, without robot/GPU."""
import numpy as np
from cable_pointcloud.cmcor_multigrasp_recorder import warp_mask


def test_translation_and_invalid_correspondence():
    mask = np.zeros((8, 8), bool)
    mask[3, 3] = True
    valid = np.ones_like(mask)
    forward = np.zeros((8, 8, 2), np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0], backward[..., 0] = 1, -1
    mapped, good = warp_mask(mask, valid, forward, backward, 0.1)
    assert mapped.sum() == 1 and mapped[3, 4]
    assert not good[:, 0].any()
    backward[3, 4, 0] = 0
    mapped, good = warp_mask(mask, valid, forward, backward, 0.1)
    assert not good[3, 4] and not mapped[3, 4]
