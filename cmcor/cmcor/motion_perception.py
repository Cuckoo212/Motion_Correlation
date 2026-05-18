"""
The main interface class to motion correlation and segmentation algorithms.
It can load and save data sequences.
It instantiates the optical flow estimator.
"""


import os
import json
import threading

import cv2
import numpy as np

from cmcor import motion_correlation
from cmcor import motion_segmentation
from cmcor import flow_predictor



def transform_point_to_point(transform, point):
    point4 = np.ones((4,1))
    point4[0:3,0] = point[:]
    res4 = transform.dot(point4)
    res4 = res4 / res4[-1,0]
    res = res4[0:3,0]
    return res


class MotionPerception():
    def __init__(
            self, gpu=False, use_dummy_flow=False,
            crop_origin=None, crop_size=None, scale_factor=None,
            focal_length=None):
        self.image_buffer = []
        self.action_buffer = []
        self.ee_point_buffer = []
        self.meters_per_pixel = None
        self.corr_imgs = None
        self.action2index = None
        self.moments = None
        self.cod_images = None
        self.a_images = None
        self.flow_std_images = None
        self.of_predictor = None
        self.use_dummy_flow = use_dummy_flow
        self.crop_origin = crop_origin
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.focal_length = focal_length
        self.gripper_depth = None
        self.poor_arm_masks = False
        if not use_dummy_flow:
            self.of_predictor = flow_predictor.FlowPredictor(gpu=gpu)
        self.perception_thread = threading.Thread()
        self.saving_thread = threading.Thread()
        self.loading_thread = threading.Thread()

    def buffer_size(self):
        return len(self.image_buffer)

    def clear_buffer(self):
        self.image_buffer.clear()
        self.action_buffer.clear()
        self.ee_point_buffer.clear()
        self.gripper_depth = None

    def add_sample(
            self, rgb_image, arm_mask, ee_point, action_index,
            depth_image=None):
        if arm_mask is None:
            image_dict = {"rgb_image": rgb_image}
        else:
            image_dict = {"rgb_image": rgb_image, "arm_mask": arm_mask}
        if depth_image is not None:
            image_dict["depth_image"] = depth_image
        self.image_buffer.append(image_dict)
        self.action_buffer.append(action_index)
        self.ee_point_buffer.append(ee_point)

    def save_buffer_to_disk(
            self, folder, grasped_cable_mask, camera_matrix=None,
            robot_base_to_camera_transform=None):
        json_path = os.path.join(folder, "actions_gripper.json")
        json_dict = {
            "action_buffer": self.action_buffer,
            "ee_point_buffer": self.ee_point_buffer,
            "crop_origin": self.crop_origin,
            "crop_size": self.crop_size,
            "scale_factor": self.scale_factor,
            }
        if camera_matrix is not None:
            json_dict["camera_matrix"] = camera_matrix.tolist()
        if robot_base_to_camera_transform is not None:
            json_dict["robot_base_to_camera_transform"] = robot_base_to_camera_transform.tolist()
        with open(json_path, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)
        grasped_cable_path = os.path.join(
            folder, "grasped_cable_00000000.png")
        if grasped_cable_mask.dtype == bool:
            cable_mask = 255*grasped_cable_mask.astype(np.uint8)
        else:
            cable_mask = grasped_cable_mask
        cv2.imwrite(grasped_cable_path, cable_mask)
        for idx, sample in enumerate(self.image_buffer):
            rgb_path = os.path.join(folder, "rgb_{:08d}.png".format(idx))
            arm_path = os.path.join(folder, "arm_{:08d}.png".format(idx))
            cv2.imwrite(rgb_path, sample["rgb_image"][...,::-1])
            cv2.imwrite(
                arm_path, 255*sample["arm_mask"].astype(np.uint8))
            if "depth_image" in sample:
                depth_path = os.path.join(
                    folder, "depth_{:08d}.png".format(idx))
                cv2.imwrite(depth_path, sample["depth_image"])

    def load_correlation_images_from_disk(self, folder):
        self.corr_imgs = []
        self.action2index = {}
        for action_idx in [0,1]:
            corr_path = os.path.join(
                folder, "corr_0_{:d}.png".format(action_idx))
            if not os.path.isfile(corr_path):
                continue
            corr_img = cv2.imread(corr_path, cv2.IMREAD_UNCHANGED)
            assert(corr_img is not None)
            corr_img = corr_img.astype(np.float32)
            corr_img = corr_img / 255.0
            self.corr_imgs.append(corr_img)
            self.action2index[action_idx] = len(self.corr_imgs) - 1

    def load_buffer_from_disk(self, folder):
        json_path = os.path.join(folder, "actions_gripper.json")
        with open(json_path, "r") as fp:
            json_dict = json.load(fp)
        self.clear_buffer()
        self.action_buffer.extend(json_dict["action_buffer"])
        self.ee_point_buffer.extend(json_dict["ee_point_buffer"])
        if "crop_origin" in json_dict:
            self.crop_origin = json_dict["crop_origin"]
        if "crop_size" in json_dict:
            self.crop_size = json_dict["crop_size"]
        if "scale_factor" in json_dict:
            self.scale_factor = json_dict["scale_factor"]
        
        if "camera_matrix" in json_dict:
            self.focal_length = json_dict["camera_matrix"][0][0]
        elif "focal_length" in json_dict:
            self.focal_length = json_dict["focal_length"]
        
        if "poor_arm_masks" in json_dict:
            self.poor_arm_masks = json_dict["poor_arm_masks"]
        else:
            self.poor_arm_masks = False
        action_indices = np.flatnonzero(
            np.array(json_dict["action_buffer"]) >= 0)
        if ("robot_base_to_camera_transform" in json_dict
                and len(action_indices) > 0):
            robot2camera = np.array(
                json_dict["robot_base_to_camera_transform"])
            gripper_point = np.array(
                json_dict["ee_point_buffer"][action_indices[0]])
            gripper_point_camera = transform_point_to_point(
                robot2camera, gripper_point)
            self.gripper_depth = gripper_point_camera[2]
        elif "gripper_depth" in json_dict:
            self.gripper_depth = json_dict["gripper_depth"]
        assert(len(self.ee_point_buffer) == len(self.action_buffer))
        self.load_correlation_images_from_disk(folder)
        idx = 0
        rgb_path = os.path.join(folder, "rgb_{:08d}.png".format(idx))
        arm_path = os.path.join(folder, "arm_{:08d}.png".format(idx))
        depth_path = os.path.join(folder, "depth_{:08d}.png".format(idx))
        while os.path.isfile(rgb_path):
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            assert(rgb_image is not None)
            rgb_image = rgb_image[...,::-1]
            image_dict = {"rgb_image": rgb_image}
            if not self.poor_arm_masks:
                arm_image = cv2.imread(arm_path, cv2.IMREAD_UNCHANGED)
                assert(arm_image is not None)
                arm_mask = arm_image > 0
                image_dict["arm_mask"] = arm_mask
            if os.path.isfile(depth_path):
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                assert(depth_image is not None)
                image_dict["depth_image"] = depth_image
            self.image_buffer.append(image_dict)
            idx += 1
            rgb_path = os.path.join(folder, "rgb_{:08d}.png".format(idx))
            arm_path = os.path.join(folder, "arm_{:08d}.png".format(idx))
            depth_path = os.path.join(folder, "depth_{:08d}.png".format(idx))
        assert(len(self.image_buffer) == len(self.action_buffer))

    def compute_segmentation_images(self, thr_moving=2.0):
        if not self.use_dummy_flow:
            assert(self.of_predictor is not None)
        action_codes = np.array(self.action_buffer)
        gripper_positions = np.array(self.ee_point_buffer)
        result = motion_segmentation.compute_segmentation_images(
            gripper_positions,
            action_codes,
            self.image_buffer,
            self.of_predictor,
            use_dummy_flow=self.use_dummy_flow,
            thr_moving=thr_moving,
            crop_origin=self.crop_origin,
            crop_size=self.crop_size,
            scale_factor=self.scale_factor)
        segmentations, action2index = result
        self.corr_imgs = segmentations
        self.action2index = action2index

    def compute_correlation_images(
            self, gripper_depth=None, focal_length=None, sample_point=None):
        if gripper_depth is None:
            gripper_depth = self.gripper_depth
        if focal_length is None:
            focal_length = self.focal_length
        assert(gripper_depth > 0)
        assert(focal_length > 0)
        if not self.use_dummy_flow:
            assert(self.of_predictor is not None)
        scale_meters_per_pixel = gripper_depth/focal_length
        action_codes = np.array(self.action_buffer)
        gripper_positions = np.array(self.ee_point_buffer)
        result = motion_correlation.compute_correlation_images(
            scale_meters_per_pixel,
            gripper_positions,
            action_codes,
            self.image_buffer,
            self.of_predictor,
            sample_point=sample_point,
            use_dummy_flow=self.use_dummy_flow,
            crop_origin=self.crop_origin,
            crop_size=self.crop_size,
            scale_factor=self.scale_factor)
        corr_imgs, action2index, moments, cod_images, a_images, flow_std_images = result
        self.corr_imgs = corr_imgs
        self.action2index = action2index
        self.moments = moments
        self.cod_images = cod_images
        self.a_images = a_images
        self.flow_std_images = flow_std_images

    def start_computation(self, gripper_depth, focal_length):
        if self.is_computing() or self.is_loading():
            return False
        self.corr_imgs = None
        self.perception_thread = threading.Thread(
            target=self.compute_correlation_images,
            args=[gripper_depth, focal_length])
        self.perception_thread.start()
        return True

    def is_computing(self):
        return self.perception_thread.is_alive()

    def is_saving(self):
        return self.saving_thread.is_alive()

    def is_loading(self):
        return self.loading_thread.is_alive()

    def start_saving(
            self, folder, grasped_cable_mask, camera_matrix=None,
            robot_base_to_camera_transform=None):
        if self.is_saving() or self.is_loading():
            return False
        self.saving_thread = threading.Thread(
            target=self.save_buffer_to_disk,
            args=[
                folder, grasped_cable_mask, camera_matrix,
                robot_base_to_camera_transform])
        self.saving_thread.start()
        return True

    def start_loading(self, folder):
        if self.is_saving() or self.is_computing() or self.is_loading():
            return False
        self.loading_thread = threading.Thread(
            target=self.load_buffer_from_disk,
            args=[folder])
        self.loading_thread.start()
        return True
