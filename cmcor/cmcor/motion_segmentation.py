"""
Compute motion segmentation images for a motion sample sequence.

- Initialize one segmentation image per action, set all pixels to zero.
- A pixel of a segmentation image eventually contains the number of motion
segmentation masks voting for that pixel.
"""

import os
import json
import pathlib
import argparse
from copy import deepcopy

import cv2
import numpy as np
import sklearn.decomposition

from cmcor.motion_correlation import get_reference_position
from cmcor.motion_correlation import inflate_arm_mask
from cmcor.motion_correlation import masked_flow
from cmcor.motion_correlation import flow_image_norm



def initialize_segmentation_images(action_index, image_shape):
    """Create empty segmentation images for each action in action_index."""
    actions = np.unique(action_index)
    if actions[0] == -1:
        actions = actions[1:]
    segmentations = [np.zeros(image_shape, dtype=np.float32) for a in actions]
    votes_per_action = [0 for a in actions]
    action2index = {action:idx for idx, action in enumerate(actions)}
    return segmentations, votes_per_action, action2index


def compute_segmentation_images(
        gripper_positions, action_codes, image_buffer,
        flow_predictor, use_dummy_flow=False, thr_moving=2.0,
        min_gripper_diff=0.001,
        crop_origin=None, crop_size=None, scale_factor=None):
    segmentations = None
    votes_per_action = None
    action2index = None
    last_action = -1
    ref_flow = None
    shape = None
    ref_gripper = None
    v_gripper = None
    last_sampled_d_gripper = None
    assert(len(action_codes) == len(image_buffer))
    assert(len(gripper_positions) == len(image_buffer))
    if not image_buffer:
        return None, None, None
    target_image_sample = image_buffer[-1]
    for idx, (gripper, action, image_sample) in enumerate(
            zip(gripper_positions, action_codes, image_buffer)):
        if action < 0:
            continue
        flow_img = None
        if action != last_action:
            sel = action_codes == action
            ref_gripper, v_gripper = get_reference_position(
                gripper_positions[sel,:])
            flow_img = masked_flow(
                flow_predictor, image_sample, target_image_sample,
                use_dummy_flow, crop_origin, crop_size, scale_factor)
            ref_flow = deepcopy(flow_img)
            last_action = action
            last_sampled_d_gripper = None
        d_gripper = (gripper - ref_gripper).dot(v_gripper)
        if last_sampled_d_gripper is None:
            last_sampled_d_gripper = d_gripper
        else:
            if abs(d_gripper - last_sampled_d_gripper) < min_gripper_diff:
                continue
            last_sampled_d_gripper = d_gripper
        if flow_img is None:
            flow_img = masked_flow(
                flow_predictor, image_sample, target_image_sample,
                use_dummy_flow, crop_origin, crop_size, scale_factor)
        if segmentations is None:
            shape = flow_img.shape[0:2]
            result = initialize_segmentation_images(
                action_codes, shape)
            segmentations, votes_per_action, action2index = result
        relative_flow = flow_img - ref_flow
        d_flow = flow_image_norm(relative_flow)
        moving_mask = d_flow > thr_moving
        segmentations[action2index[action]][moving_mask] += 1
        votes_per_action[action2index[action]] += 1
    for seg_img, n_votes in zip(segmentations, votes_per_action):
        if n_votes == 0:
            continue
        seg_img[...] = (100*seg_img[...]) / n_votes
    return segmentations, action2index
