"""
Motion correlation on a sequence of images, gripper positions, action labels.
"""

import os
import json
import time
import pathlib
import argparse
import threading
import queue
from copy import deepcopy
from functools import lru_cache

import cv2
import numpy as np
import sklearn.decomposition

from cmcor import motion_perception_settings



def get_metric_flow(flow_img, meters_per_pixel):
    return flow_img * meters_per_pixel


def initialize_moment_images(action_index, image_shape):
    """Create empty moment images for each action in action_index."""
    keys = ["sum_f", "sum_g", "sum_f2", "sum_g2", "sum_fg"]
    actions = np.unique(action_index)
    if actions[0] == -1:
        actions = actions[1:]
    moments = [{} for i in range(actions.size)]
    action2index = {action:idx for idx, action in enumerate(actions)}
    for m in moments:
        for k in keys:
            m[k] = np.zeros(image_shape, dtype=np.float64)
        m["n_samples"] = 0
        m["d_flow_samples"] = []
        m["d_gripper_samples"] = []
        m["d_manual"] = []
    return moments, action2index


def update_moments(
        action_moments, d_gripper, d_flow, sample_point=None,
        d_manual=None, print_timing=False):
    """Update the moments of d_gripper and d_flow in action_moments."""
    #assert(np.all(np.isfinite(d_flow)))
    #assert(np.all(np.isfinite(d_gripper)))
    t_0 = time.time()
    action_moments["sum_f"] += d_flow
    action_moments["sum_f2"] += d_flow**2
    action_moments["sum_g"] += d_gripper
    action_moments["sum_g2"] += d_gripper**2
    action_moments["sum_fg"] += d_gripper*d_flow
    action_moments["n_samples"] += 1
    if sample_point is not None:
        action_moments["d_flow_samples"].append(
            d_flow[sample_point[0], sample_point[1]])
        action_moments["d_gripper_samples"].append(d_gripper)
    if d_manual is not None:
        action_moments["d_manual"].append(d_manual)
    t_1 = time.time()
    if print_timing:
        print(f"    update_moments took {t_1-t_0:.6f} s")


def correlations_from_moments(moments):
    """Return a sequence of correlation images computed from moments.

    https://en.wikipedia.org/wiki/Correlation#Sample_correlation_coefficient
    """
    corr_imgs = []
    for action in moments:
        n = action["n_samples"]
        corr = n*action["sum_fg"] - action["sum_f"]*action["sum_g"]
        denom = np.sqrt(n*action["sum_f2"] - action["sum_f"]**2)
        denom = denom * np.sqrt(n*action["sum_g2"] - action["sum_g"]**2)
        valid = denom > 1e-6
        corr[valid] = corr[valid] / denom[valid]
        corr[np.logical_not(valid)] = np.nan
        corr_imgs.append(corr)
    return corr_imgs


def linear_cod_from_moments(
        moments, min_a, min_cod, print_timing=False):
    """Return a sequence of linear coefficient of determination (CoD) images.

    Fit a linear model f(g) = a*g + b to the pairs of f (flow) and g (gripper)
    displacements. Compute the R squared (coefficient of determination) value
    measuring the goodness of fit. Compute it from moments.

    For pixels where |a| < min_a, set CoD to np.nan.
    For pixels where CoD < min_cod, set CoD to np.nan.

    https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    t_0 = time.time()
    filtered_cod_imgs = []
    cod_images = []
    a_images = []
    flow_std_images = []
    for action in moments:
        #assert(np.all(action["sum_f2"] >= 0))
        #assert(np.all(action["sum_g2"] >= 0))
        n = action["n_samples"]
        assert(n > 0)
        det = n*action["sum_g2"] - action["sum_g"]**2
        abs_det = np.abs(det)
        var_f = (1.0/n)*action["sum_f2"] - ((1.0/n)*action["sum_f"])**2
        std_f = np.sqrt(var_f)
        flow_std_images.append(deepcopy(std_f))
        #assert(np.all(var_f >= 0))
        valid = np.logical_and(abs_det > 1e-6, var_f > 1e-8)
        a_nom = n*action["sum_fg"] - action["sum_g"]*action["sum_f"]
        b_nom = (-action["sum_g"]*action["sum_fg"]
            + action["sum_g2"]*action["sum_f"])
        a_valid = a_nom[valid]/det[valid]
        b_valid = b_nom[valid]/det[valid]
        cod = np.zeros(action["sum_f2"].shape, dtype=np.float64)
        cod[...] = np.nan
        cod[valid] = action["sum_f2"][valid]
        cod[valid] = cod[valid] - 2*a_valid*action["sum_fg"][valid]
        cod[valid] = cod[valid] - 2*b_valid*action["sum_f"][valid]
        cod[valid] = cod[valid] + (a_valid**2)*action["sum_g2"][valid]
        cod[valid] = cod[valid] + 2*a_valid*b_valid*action["sum_g"][valid]
        cod[valid] = cod[valid] + n*b_valid**2
        values = cod[valid]
        assert(np.all(cod[valid] >= 0))
        cod[valid] = 1.0 - cod[valid]/(n*var_f[valid])
        cod_images.append(deepcopy(cod))
        a_image = np.zeros(cod.shape, dtype=np.float64)
        a_image[...] = np.nan
        a_image[valid] = a_valid
        a_images.append(a_image)
        valid_num = np.flatnonzero(valid)
        filter_out = np.logical_or(
            np.abs(a_valid) < min_a, cod[valid] < min_cod)
        valid_num_weak = valid_num[filter_out]
        cod.flat[valid_num_weak] = np.nan
        filtered_cod_imgs.append(cod)
    t_1 = time.time()
    if print_timing:
        print(f"    linear_cod_from_moments took {t_1-t_0:.6f} s")
    return filtered_cod_imgs, cod_images, a_images, flow_std_images


def get_reference_position(gripper_position):
    #ref_pos = np.mean(gripper_position, axis=0)
    ref_pos = gripper_position[0,:]
    pca = sklearn.decomposition.PCA(n_components=1)
    pca.fit(gripper_position - ref_pos[None,:])
    vect = pca.components_[0,:]
    assert(abs(np.linalg.norm(vect) - 1.0) < 1e-4)
    return ref_pos, vect


def update_v_flow(v_flow, relative_flow, d_flow, thr_moving):
    sel = np.logical_and(
        np.abs(d_flow) > thr_moving,
        np.isnan(v_flow[...,0]))
    v_flow[sel,:] = relative_flow[sel,:]/d_flow[sel,None]


def initialize_cov_images(image_shape):
    """Create empty covariance matrix element images."""
    keys = ["sum_x2", "sum_y2", "sum_xy"]
    cov = {}
    for k in keys:
        cov[k] = np.zeros(image_shape, dtype=np.float64)
    return cov


def update_flow_cov(cov, relative_flow, mask):
    """Update the elements of the flow covariance matrices."""
    cov["sum_x2"][mask] += relative_flow[mask,0]**2
    cov["sum_y2"][mask] += relative_flow[mask,1]**2
    cov["sum_xy"][mask] += relative_flow[mask,0]*relative_flow[mask,1]


def flow_image_norm(image):
    prod = image**2
    return np.sqrt(prod[...,0] + prod[...,1])


def update_v_flow_pca(
        v_flow, relative_flow, cov, moving_mask, print_timing=False):
    t_0 = time.time()
    flow_shape = relative_flow.shape
    a = cov["sum_x2"]
    b = cov["sum_xy"]
    c = b
    d = cov["sum_y2"]
    det = (a+d)**2 - 4*(a*d-b*c)
    valid = np.logical_and(det >= 0, moving_mask)
    l_max_valid = 0.5*(a[valid]+d[valid]+np.sqrt(det[valid]))
    v1 = np.zeros(flow_shape, dtype=np.float32)
    v2 = np.zeros(flow_shape, dtype=np.float32)
    v1[valid,0] = l_max_valid-d[valid]
    v1[valid,1] = c[valid]
    v2[valid,0] = b[valid]
    v2[valid,1] = l_max_valid-a[valid]
    v1_norm = flow_image_norm(v1)
    v2_norm = flow_image_norm(v2)
    v2_sel = v2_norm > v1_norm
    v1_norm[v2_sel] = v2_norm[v2_sel]
    v1[v2_sel,:] = v2[v2_sel,:]
    valid = np.logical_and(valid, v1_norm > 1e-10)
    v_flow[valid,:] = v1[valid,:]/v1_norm[valid,None]
    t_1 = time.time()
    if print_timing:
        print(f"    update_v_flow_pca took {t_1-t_0:.6f} s")


def compute_d_flow_pca(
        flow_metric, ref_flow, v_flow, flow_cov, thr_static, thr_moving,
        print_timing=False):
    t_0 = time.time()
    relative_flow = flow_metric - ref_flow
    d_flow = flow_image_norm(relative_flow)
    moving_mask = d_flow > thr_moving
    update_flow_cov(flow_cov, relative_flow, moving_mask)
    update_v_flow_pca(v_flow, relative_flow, flow_cov, moving_mask)
    v_valid = np.isfinite(v_flow[...,0])
    sel_product = v_flow[v_valid,:]*relative_flow[v_valid,:]
    d_flow[v_valid] = sel_product[...,0] + sel_product[...,1]
    d_flow[np.logical_not(v_valid)] = 0
    d_flow[np.abs(d_flow) < thr_static] = 0
    t_1 = time.time()
    if print_timing:
        print(f"    compute_d_flow_pca took {t_1-t_0:.6f} s")
    return d_flow


def compute_d_flow(flow_metric, ref_flow, v_flow, thr_static, thr_moving):
    relative_flow = flow_metric - ref_flow
    d_flow = flow_image_norm(relative_flow)
    update_v_flow(v_flow, relative_flow, d_flow, thr_moving)
    v_valid = np.isfinite(v_flow[...,0])
    sel_product = v_flow[v_valid,:]*relative_flow[v_valid,:]
    d_flow[v_valid] = np.sum(sel_product, axis=1)
    d_flow[np.abs(d_flow) < thr_static] = 0
    return d_flow


def inflate_arm_mask(mask):
    dilation_shape = cv2.MORPH_RECT
    dilation_size = 5
    element = cv2.getStructuringElement(dilation_shape,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size))
    dilated = cv2.dilate(255*(mask).astype(np.uint8), element)
    return dilated>0


def crop_and_scale(image, crop_origin, crop_size, scale_factor):
    if crop_origin is None or crop_size is None:
        cropped = image
    else:
        cropped = image[
            crop_origin[1]:crop_origin[1]+crop_size[1],
            crop_origin[0]:crop_origin[0]+crop_size[0],:]
    if scale_factor is None:
        cropped_scaled = cropped
    else:
        cropped_scaled = cv2.resize(
            cropped, (0,0), fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_CUBIC)
    return cropped_scaled


def descale_and_decrop(
        image, crop_origin, crop_size, scale_factor, orig_size,
        is_flow_image=False):
    if scale_factor is None:
        descaled = image
    else:
        if is_flow_image:
            image = image * (1.0/scale_factor)
        descaled = cv2.resize(
            image, (0,0), fx=1.0/scale_factor, fy=1.0/scale_factor,
            interpolation=cv2.INTER_CUBIC)
    if crop_origin is None or crop_size is None:
        full_image = descaled
    else:
        full_image = np.zeros(
            (orig_size[0], orig_size[1], descaled.shape[2]),
            dtype=descaled.dtype)
        full_image[
            crop_origin[1]:crop_origin[1]+crop_size[1],
            crop_origin[0]:crop_origin[0]+crop_size[0],:] = descaled
    return full_image


@lru_cache(maxsize=1)
def precompute_grid_shape(shape):
    """Return a 2D meshgrid (its v, u coordinates) for an image shape."""
    ui = np.arange(0, shape[1])
    vi = np.arange(0, shape[0])
    v,u = np.meshgrid(vi, ui, indexing='ij')
    grid = np.zeros((shape[0],shape[1],2))
    grid[...,0] = v
    grid[...,1] = u
    return grid


def filter_flow(flow, reference_mask):
    assert(flow.shape[2] == 2)
    grid = precompute_grid_shape((flow.shape[0], flow.shape[1]))
    # flow(y,x) = d_horizontal, d_vertical
    # grid(y,x) = d_vertical (v), d_horizontal (u)
    ref_points = flow[...,::-1] + grid
    ref_points = ref_points.astype(int)
    ref_points[...,0] = np.clip(
        ref_points[...,0], a_min=0, a_max=flow.shape[0]-1)
    ref_points[...,1] = np.clip(
        ref_points[...,1], a_min=0, a_max=flow.shape[1]-1)
    to_erase = reference_mask[ref_points[...,0], ref_points[...,1]]
    flow[to_erase,:] = 0
    return flow


def masked_flow(
        flow_predictor, image_sample, target_image_sample, use_dummy_flow,
        crop_origin, crop_size, scale_factor, print_timing=False,
        do_filter_flow=False):

    if use_dummy_flow:
        full_flow_img = np.zeros(
            (image_sample["rgb_image"].shape[0],
                image_sample["rgb_image"].shape[1], 2))
    else:
        target = target_image_sample["rgb_image"]
        reference = image_sample["rgb_image"]
        if "rgb_image_cropped_scaled" not in target_image_sample:
            target_image_sample["rgb_image_cropped_scaled"] = crop_and_scale(
                target, crop_origin, crop_size, scale_factor)
        target_cs = target_image_sample["rgb_image_cropped_scaled"]
        reference_cs = crop_and_scale(
            reference, crop_origin, crop_size, scale_factor)
        t_0 = time.time()
        flow_img = flow_predictor.flow(target_cs, reference_cs)
        t_1 = time.time()
        if print_timing:
            print(f"    masked_flow took {t_1-t_0:.4f} s per image pair")

        orig_size = [target.shape[0], target.shape[1], flow_img.shape[2]]
        full_flow_img = descale_and_decrop(
            flow_img, crop_origin, crop_size, scale_factor, orig_size,
            is_flow_image=True)
    if do_filter_flow and "arm_mask" in image_sample:
        arm_mask_ref = inflate_arm_mask(image_sample["arm_mask"])
        full_flow_img = filter_flow(full_flow_img, arm_mask_ref)
        #arm_mask = inflate_arm_mask(target_image_sample["arm_mask"])
        #full_flow_img[arm_mask,:] = 0
    return full_flow_img


def masked_metric_flow(
        flow_predictor, image_sample, target_image_sample, use_dummy_flow,
        crop_origin, crop_size, scale_factor,
        scale_meters_per_pixel):
    flow_img = masked_flow(
        flow_predictor, image_sample, target_image_sample, use_dummy_flow,
        crop_origin, crop_size, scale_factor)
    flow_metric = get_metric_flow(flow_img, scale_meters_per_pixel)
    return flow_metric


def masked_metric_flow_task(
        queue_out, input_samples,
        flow_predictor, target_image_sample, use_dummy_flow,
        crop_origin, crop_size, scale_factor, scale_meters_per_pixel):
    for sample in input_samples:
        if "flow_metric" in sample:
            queue_out.put(sample)
            continue
        image = sample["image"]
        flow_metric = masked_metric_flow(
                flow_predictor, image, target_image_sample,
                use_dummy_flow, crop_origin, crop_size, scale_factor,
                scale_meters_per_pixel)
        sample["flow_metric"] = flow_metric
        queue_out.put(sample)
    queue_out.put(None)


def d_flow_moments_task(
        queue_in, action, ref_flow, v_flow, flow_cov, moments, action2index,
        sample_point,
        thr_static_px, thr_moving_pca_px, scale_meters_per_pixel,
        print_timing=False):
    while True:
        sample = queue_in.get()
        if sample is None:
            break
        flow_metric = sample["flow_metric"]
        d_manual = sample["d_manual"]
        d_gripper = sample["d_gripper"]
        t_0 = time.time()
        d_flow = compute_d_flow_pca(
            flow_metric, ref_flow, v_flow, flow_cov,
            thr_static_px*scale_meters_per_pixel,
            thr_moving_pca_px*scale_meters_per_pixel)
        update_moments(
            moments[action2index[action]], d_gripper, d_flow,
            sample_point, d_manual)
        t_1 = time.time()
        if print_timing:
            print(f"    d_flow_moments_task took {t_1-t_0:.4f} s per image")



def run_pipeline(
        input_samples, action, ref_flow, v_flow, flow_cov, moments,
        action2index, sample_point, flow_predictor, target_image_sample,
        use_dummy_flow, thr_static_px, thr_moving_pca_px,
        crop_origin, crop_size, scale_factor, scale_meters_per_pixel):
    thread_flow = None
    thread_moments = None
    queue_flow = queue.Queue()

    thread_flow = threading.Thread(
        target=masked_metric_flow_task,
        args=[queue_flow, input_samples,
            flow_predictor, target_image_sample, use_dummy_flow,
            crop_origin, crop_size, scale_factor,
            scale_meters_per_pixel])
    thread_moments = threading.Thread(
        target=d_flow_moments_task,
        args=[queue_flow, action, ref_flow, v_flow, flow_cov,
            moments, action2index, sample_point,
            thr_static_px, thr_moving_pca_px,
            scale_meters_per_pixel])
    thread_flow.start()
    thread_moments.start()
    thread_flow.join()
    thread_moments.join()


def compute_correlation_images(
        scale_meters_per_pixel, gripper_positions, action_codes, image_buffer,
        flow_predictor, sample_point=None, x_manual=None,
        min_gripper_diff=0.001, use_dummy_flow=False,
        crop_origin=None, crop_size=None, scale_factor=None):
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    thr_static_px = 0.67
    thr_moving_pca_px = 1.33 # 2.0

    input_samples = []

    moments = None
    action2index = None
    last_action = -1
    ref_gripper = None
    v_gripper = None
    ref_flow = None
    v_flow = None
    ref_manual = None
    flow_cov = None
    shape = None
    last_sampled_d_gripper = None
    assert(len(action_codes) == len(image_buffer))
    assert(len(gripper_positions) == len(image_buffer))
    if not image_buffer:
        return None, None, None, None, None, None
    target_image_sample = image_buffer[-1]
    for idx, (gripper, action, image_sample) in enumerate(
            zip(gripper_positions, action_codes, image_buffer)):
        if action < 0:
            continue
        flow_metric = None
        if action != last_action:
            if input_samples:
                assert(moments is not None)
                run_pipeline(
                    input_samples, last_action, ref_flow, v_flow, flow_cov,
                    moments, action2index, sample_point, flow_predictor,
                    target_image_sample, use_dummy_flow,
                    thr_static_px, thr_moving_pca_px, crop_origin,
                    crop_size, scale_factor, scale_meters_per_pixel)
            flow_metric = masked_metric_flow(
                flow_predictor, image_sample, target_image_sample,
                use_dummy_flow, crop_origin, crop_size, scale_factor,
                scale_meters_per_pixel)
            shape = flow_metric.shape[0:2]
            flow_cov = initialize_cov_images(shape)
            sel = action_codes == action
            ref_gripper, v_gripper = get_reference_position(
                gripper_positions[sel,:])
            ref_flow = deepcopy(flow_metric)
            v_flow = np.zeros(flow_metric.shape, dtype=np.float32)
            v_flow[...] = np.nan
            last_action = action
            if x_manual is not None:
                ref_manual = x_manual[idx]
            last_sampled_d_gripper = None
            input_samples = []
        sample = {}

        d_gripper = (gripper - ref_gripper).dot(v_gripper)
        if last_sampled_d_gripper is None:
            last_sampled_d_gripper = d_gripper
        else:
            if abs(d_gripper - last_sampled_d_gripper) < min_gripper_diff:
                continue
            last_sampled_d_gripper = d_gripper
        sample["d_gripper"] = d_gripper
        sample["image"] = image_sample
        if flow_metric is not None:
            sample["flow_metric"] = flow_metric
        if x_manual is not None:
            d_manual = x_manual[idx] - ref_manual
        else:
            d_manual = None
        sample["d_manual"] = d_manual
        if moments is None:
            assert(flow_metric is not None)
            shape = flow_metric.shape[0:2]
            moments, action2index = initialize_moment_images(
                action_codes, shape)
        input_samples.append(sample)
    if moments is None:
        return None, None, None, None, None, None
    if input_samples:
        run_pipeline(
            input_samples, last_action, ref_flow, v_flow, flow_cov,
            moments, action2index, sample_point, flow_predictor,
            target_image_sample, use_dummy_flow,
            thr_static_px, thr_moving_pca_px, crop_origin,
            crop_size, scale_factor, scale_meters_per_pixel)
    result = linear_cod_from_moments(
        moments, min_a=mp_settings.motion_correlation_min_a,
        min_cod=mp_settings.motion_correlation_min_cod)
    corr_imgs, cod_images, a_images, flow_std_images = result
    return corr_imgs, action2index, moments, cod_images, a_images, flow_std_images
