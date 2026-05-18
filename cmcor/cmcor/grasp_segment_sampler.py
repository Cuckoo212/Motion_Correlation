"""
Sample cable segments likely suitable for grasping.
"""
import os
import sys
import time
import warnings

import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
QtWidgets.QApplication(sys.argv)
import cv2
import matplotlib.pyplot as plt

from cmcor import pixel_segment
from cmcor import cable_pose



def plane_vector_angle(plane_normal, vector):
    v_norm = np.linalg.norm(vector)
    n_norm = np.linalg.norm(plane_normal)
    angle = np.arccos(plane_normal.dot(vector)/(n_norm*v_norm))
    angle = np.pi/2 - angle
    return angle


def pixels_to_points_3d(pixels, depth, camera_matrix):
    """Return a metric 3D point for each 2D pixel.

    Arguments:
    pixels - pixel coordinates
    depth - depth image, depth in millimeters
    camera_matrix - camera matrix

    Returns:
    pixels_sel - 2D pixel coordinates with valid depth
        - pixels_sel.shape[1] == 2
    xyz - 3D points with coordinates in meters
        - xyz.shape[0] == pixels_sel.shape[0]
        - xyz.shape[1] == 3
    """
    #print(depth.shape)
    # return numpy array (N, 3)
    z_sel = depth[pixels[:,0], pixels[:,1]].flatten()
    valid = z_sel > 0
    z_sel = z_sel[valid]
    pixels_sel = pixels[valid,:]
    n_sel = np.ones(z_sel.size)
    z_sel = z_sel.astype(np.float32) / 1000.0
    u_sel = pixels_sel[:,1]
    v_sel = pixels_sel[:,0]
    uvn = np.concatenate(
        (u_sel[None,:], v_sel[None,:], n_sel[None,:]), axis=0)
    K_inv = np.linalg.inv(camera_matrix)
    xyn = K_inv.dot(uvn)
    xyn[0:2,:] = xyn[0:2,:]/xyn[None,2,:]
    xyz = xyn * z_sel[None,:]
    xyz = xyz.T
    assert(xyz.shape[0] == pixels_sel.shape[0])
    assert(xyz.shape[1] == 3)
    assert(pixels_sel.shape[1] == 2)
    return pixels_sel, xyz


class GraspSegmentSampler():

    def __init__(
            self, n_samples_per_distance=16,
            max_cable_image_angle=np.pi/4):
        self.grasp_distance_thresholds = [0.2, 0.15, 0.1]
        self.n_samples_per_distance = n_samples_per_distance
        self.n_sampled_in_distance = 0
        self.current_distance_idx = 0
        #self.min_distance_between_grasps = 0.2
        self.max_cable_image_angle = max_cable_image_angle
        self.rng = np.random.default_rng()
        self.previous_grasps = []

    def set_input_images(self, depth, camera_matrix, vote_image):
        self.depth = depth
        self.camera_matrix = camera_matrix
        self.vote_image = vote_image
        self.blacklist_mask = np.zeros(self.vote_image.shape, dtype=bool)
        self.n_sampled_in_distance = 0
        self.current_distance_idx = 0
        self.current_vote_group = np.amax(self.vote_image)
        self.update_vote_group_mask()
        self.update_min_distance_between_grasps(
            self.grasp_distance_thresholds[self.current_distance_idx])

    def update_vote_group_mask(self):
        self.sampling_pool_mask = self.vote_image >= self.current_vote_group
        self.sampling_pool_mask[self.blacklist_mask] = False
        sampling_pool_mask_pixels = np.argwhere(self.sampling_pool_mask)
        if sampling_pool_mask_pixels.size == 0:
            self.sampling_pool_pixels = sampling_pool_mask_pixels
            self.sampling_pool_points_3d = np.array([])
            return
        pixels_sel, xyz = pixels_to_points_3d(
            sampling_pool_mask_pixels, self.depth, self.camera_matrix)
        self.sampling_pool_pixels = pixels_sel
        self.sampling_pool_points_3d = xyz
        dists = np.linalg.norm(
            (self.previous_grasps_array[None,:,:]
            - self.sampling_pool_points_3d[:,None,:]), axis=2)
        self.min_point_grasp_distances = np.amin(dists, axis=1)
    
    def update_min_distance_between_grasps(self, min_distance):
        if self.sampling_pool_points_3d.size == 0:
            return
        sel = self.min_point_grasp_distances > min_distance
        self.far_enough_pixels = self.sampling_pool_pixels[sel,:]
        self.far_enough_points_3d = self.sampling_pool_points_3d[sel,:]

    def add_previous_grasp(self, center):
        self.previous_grasps.append(center)
        self.previous_grasps_array = np.array(self.previous_grasps)

    def clear_previous_grasps(self):
        self.previous_grasps = []
        self.previous_grasps_array = np.array(self.previous_grasps)

    def nearest_grasp_distance(self, center):
        distances = np.linalg.norm(
            self.previous_grasps_array - center[None,:], axis=1)
        min_dist = np.amin(distances)
        return min_dist

    def blacklist_sample(self, sampled_result):
        assert(sampled_result is not None)
        grasp, grasp_mask = sampled_result
        mask = grasp_mask
        if mask.dtype != bool:
            mask = mask > 0
        self.blacklist_mask[mask] = True
        self.update_vote_group_mask()
        self.update_min_distance_between_grasps(
            self.grasp_distance_thresholds[self.current_distance_idx])

    def grasp_from_pixel(self, pixel):
        grasp_mask = pixel_segment.pixel2segment(
            pixel, self.sampling_pool_mask)
        if grasp_mask is None:
            return None
        grasp = cable_pose.find_cable_pose_depth(
            self.depth, self.camera_matrix, grasp_mask)
        if grasp is None:
            return None
        center, axis, points = grasp
        #dst = self.nearest_grasp_distance(center)
        #print("nearest grasp distance", dst, "m")
        #if dst < self.min_distance_between_grasps:
        #    return None
        image_plane_normal = np.array([0,0,1])
        angle = plane_vector_angle(image_plane_normal, axis)
        if angle > self.max_cable_image_angle:
            return None
        return grasp, grasp_mask

    def sample_once(self):
        if self.far_enough_pixels.size == 0:
            return None
        pixel_idx = self.rng.integers(0, self.far_enough_pixels.shape[0])
        pixel = self.far_enough_pixels[pixel_idx,:]
        point_3d = self.far_enough_points_3d[pixel_idx,:]
        return self.grasp_from_pixel(pixel)
    
    def sample_in_distance(self):
        result = None
        while (result is None
               and self.n_sampled_in_distance < self.n_samples_per_distance):
            result = self.sample_once()
            self.n_sampled_in_distance += 1
        if result is not None:
            self.blacklist_sample(result)
        return result
    
    def sample_in_vote_group(self, debug=False):
        """
        Gradually decrease the grasp distance threshold."
        """
        warnings.warn(
            ("sample_in_vote_group is deprecated. "
            "Use sample_in_vote_group_ordered instead."), DeprecationWarning)
        result = None
        while (result is None
                and self.current_distance_idx
                < len(self.grasp_distance_thresholds)):
            result = self.sample_in_distance()
            if debug:
                sampling_distance = self.grasp_distance_thresholds[
                    self.current_distance_idx]
                print(
                    f"sample_in_distance success {result is not None}, "
                    f"samples in distance {self.n_sampled_in_distance}, "
                    f"distance {sampling_distance} m, "
                    f"vote group {self.current_vote_group}")
            if result is None:
                self.current_distance_idx += 1
                if (self.current_distance_idx
                        >= len(self.grasp_distance_thresholds)):
                    break
                self.update_min_distance_between_grasps(
                    self.grasp_distance_thresholds[self.current_distance_idx])
                self.n_sampled_in_distance = 0
        return result

    def sample_in_vote_group_ordered(self, debug=False):
        """
        Return the most distant segments first.
        """
        # TODO: Who resets self.n_sampled_in_distance = 0 when vote group sampling fails?
        result = None
        if self.sampling_pool_pixels.size == 0:
            return None
        min_exploration_distance = self.grasp_distance_thresholds[-1]
        sort_idx = np.argsort(self.min_point_grasp_distances)
        sort_idx = sort_idx[::-1] # Sort from max to min distance.
        sorted_pixels = self.sampling_pool_pixels[sort_idx,:]
        sorted_distances = self.min_point_grasp_distances[sort_idx]
        sel = sorted_distances > min_exploration_distance
        sorted_pixels = sorted_pixels[sel,:]
        pixel_idx = 0
        for pixel in sorted_pixels:
            result = self.grasp_from_pixel(pixel)
            self.n_sampled_in_distance += 1
            if debug:
                print(
                    f"sample_ordered success {result is not None}, "
                    f"samples in distance {self.n_sampled_in_distance}, "
                    f"vote group {self.current_vote_group}")
            if result is not None:
                self.blacklist_sample(result)
                break
        return result

    def sample_grasp(self):
        """The high-level grasp sampling function. Returns a grasp or None.

        Sampling hierarchy: vote group -> distance threshold -> samples
        """
        result = None
        while result is None and self.current_vote_group > 0:
            #result = self.sample_in_vote_group()
            result = self.sample_in_vote_group_ordered()
            if result is None:
                self.current_vote_group -= 1
                if self.current_vote_group <= 0:
                    break
                self.update_vote_group_mask()
                self.current_distance_idx = 0
                self.update_min_distance_between_grasps(
                    self.grasp_distance_thresholds[self.current_distance_idx])
        return result

    def reset_sampler_simple(
            self, depth, camera_matrix, motion_masks, last_grasp_center,
            do_clear_grasps=True):
        """Simply setup or reset the sampler with new motion_masks."""
        if do_clear_grasps:
            self.clear_previous_grasps()
        self.add_previous_grasp(last_grasp_center)
        vote_image = None
        for mask in motion_masks:
            mask = mask > 0
            if vote_image is None:
                vote_image = mask.astype(np.float32)
            else:
                vote_image[mask] += 1
        self.set_input_images(depth, camera_matrix, vote_image)



def load_vote_image(image_root):
    vote_image = None
    for name in sorted(os.listdir(image_root)):
        if not name.startswith("corr_"):
            continue
        mask_path = os.path.join(image_root, name)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_image = mask_image > 0
        if vote_image is None:
            vote_image = mask_image.astype(np.float32)
        else:
            vote_image[mask_image] += 1
    return vote_image


def quantity_stats(v):
    """Return several basic statistics with labels for the values in v."""
    labels = ["mean","median","min","q0.05","q0.1","q0.2",
        "q0.8","q0.9","q0.95","max"]
    stats = [np.nanmean(v),np.nanmedian(v),np.nanmin(v),
        np.nanquantile(v,0.05),np.nanquantile(v,0.1),
        np.nanquantile(v,0.2),np.nanquantile(v,0.8),
        np.nanquantile(v,0.9),np.nanquantile(v,0.95),
        np.nanmax(v)]
    return labels, stats

def print_scalar_stats(numbers):
    labels, stats = quantity_stats(numbers)
    for label, value in zip(labels, stats):
        print(label, "{:.6f}".format(value))


def test_repeat_sample_once(sampler):
    n_trials = 10000
    n_failures = 0
    durations = []
    for n in range(n_trials):
        t_0 = time.time()
        #result = sampler.sample_grasp()
        result = sampler.sample_once()
        t_1 = time.time()
        durations.append(t_1-t_0)
        if result is None:
            n_failures += 1
    print("Duration of a sampling in seconds:")
    print_scalar_stats(durations)
    print("Sampling failure ratio: {:.2f} %.".format(100*n_failures/n_trials))
    return result


def get_camera_matrix():
    camera_matrix = np.array(
        [[644.8648, 0.0, 645.5546],
         [0.0, 643.8187, 363.1338],
         [0.0, 0.0, 1.0]])
    return camera_matrix


def setup_sampler_for_test(sampler, image_root):
    camera_matrix = get_camera_matrix()
    reference_depth = 1.1   # meters
    print("============ Test starts. ============")
    print(image_root)
    grasp_path = os.path.join(image_root, "grasped_0.png")
    grasp_image = cv2.imread(grasp_path, cv2.IMREAD_UNCHANGED)
    depth = np.zeros(grasp_image.shape, dtype=np.uint16)
    depth[...] = int(reference_depth*1000)
    grasp = cable_pose.find_cable_pose_depth(
        depth, camera_matrix, grasp_image)
    assert(grasp is not None)
    grasp_center, grasp_axis, grasp_points = grasp
    sampler.clear_previous_grasps()
    sampler.add_previous_grasp(grasp_center)
    vote_image = load_vote_image(image_root)
    sampler.set_input_images(depth, camera_matrix, vote_image)
    return vote_image


def test_sampler_failure_rate(sampler, image_root):
    vote_image = setup_sampler_for_test(sampler, image_root)
    for vote_group in [1,2]:
        sampler.current_vote_group = vote_group
        sampler.update_vote_group_mask()
        print(f"---Vote group {sampler.current_vote_group}---")
        for min_distance in sampler.grasp_distance_thresholds:
            print(f"--Min. distance {min_distance}--")
            sampler.update_min_distance_between_grasps(min_distance)
            result = test_repeat_sample_once(sampler)


def test_sampler_show(sampler, vote_image, expect_success=True):
    t_0 = time.time()
    result = sampler.sample_grasp()
    t_1 = time.time()
    print("Sampling took {:.6f} seconds.".format(t_1-t_0))
    if expect_success is None:
        if result is not None:
            sampled_grasp, sampled_grasp_mask = result
            plt.imshow(vote_image)
            plt.imshow(sampled_grasp_mask, alpha=0.7)
            plt.gcf().set_size_inches(12,7)
            plt.show()
    elif expect_success:
        assert(result is not None)
        sampled_grasp, sampled_grasp_mask = result
        plt.imshow(vote_image)
        plt.imshow(sampled_grasp_mask, alpha=0.7)
        plt.gcf().set_size_inches(12,7)
        plt.show()
    else:
        assert(result is None)
    return t_1-t_0


def main_test_failure_rate():
    dataset_root = "test_data/mcor_outputs"
    experiments = [
        "2024-08-06-113044",
        "2024-08-06-125529",
        "2024-08-06-131449",
        "2024-08-06-132157",
        "2024-08-06-122035",
        "2024-08-06-123622",
        "2024-08-06-124716",
        ]
    sampler = GraspSegmentSampler()
    for experiment in experiments:
        image_root = os.path.join(dataset_root, experiment)
        test_sampler_failure_rate(sampler, image_root)


def main_test_show():
    dataset_root = "test_data/mcor_outputs"
    experiments = [
        "2024-08-06-113044", #
        "2024-08-06-125529",
        "2024-08-06-131449",
        "2024-08-06-132157",
        "2024-08-06-122035",
        "2024-08-06-123622", #
        "2024-08-06-124716",
        ]
    output_names = [
        "z-shaped-01",
        "z-shaped-02",
        "z-shaped-03",
        "z-shaped-04",
        "d-shaped-01",
        "d-shaped-02",
        "d-shaped-03",
        ]
    expect_success = [
        [True,True,True],
        [True,True,True],
        [True,True,True],
        [True,True,True],
        [True,True,True],
        [True,True,None],
        [False,False,False],
        ]
    sampler = GraspSegmentSampler()
    total_sampling_duration = 0
    for experiment, should_succeed_list in zip(experiments, expect_success):
        image_root = os.path.join(dataset_root, experiment)
        vote_image = setup_sampler_for_test(sampler, image_root)
        for should_succeed in should_succeed_list:
            duration = test_sampler_show(sampler, vote_image, should_succeed)
            total_sampling_duration += duration
    print(f"Total sampling duration: {total_sampling_duration} seconds.")

if __name__=="__main__":
    main_test_show()
    #main_test_failure_rate()
    
