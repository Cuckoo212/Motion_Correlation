"""
Run motion correlation/segmentation or view its results on image sequence(s).
A command line interface (CLI) tool.
"""

import os
import time
import argparse
from copy import deepcopy
import shutil
import pathlib

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cmcor import motion_perception
from cmcor import motion_perception_settings



def image_to_uint8(float_image, max_value=2.0, multiplier=100):
    assert(np.all(float_image >= 0))
    float_image[float_image > max_value] = max_value
    uint8_image = (float_image*multiplier).astype(np.uint8)
    return uint8_image


MOTION_PERCEPTION_ALGORITHMS = ["correlation", "segmentation"]

class MotionPerceptionMulti():
    def __init__(
            self, folder_out, sequence_folders, gpu,
            algorithm, gripper_depth=1.1, focal_length=644.3,
            crop_origin=None, crop_size=None, scale_factor=None, mp=None):
        """
        gripper_depth = 1.1  # in meters, gripper depth in the camera frame
        focal_length = 644.3  # in pixels
        """
        assert(algorithm in MOTION_PERCEPTION_ALGORITHMS)
        use_dummy_flow = False
        if mp is None:
            self.mp = motion_perception.MotionPerception(
                gpu, use_dummy_flow, crop_origin, crop_size, scale_factor)
        else:
            self.mp = mp
        self.folder_out = folder_out
        self.sequence_folders = sequence_folders
        self.algorithm = algorithm
        mp_settings = motion_perception_settings.MotionPerceptionSettings()
        self.corr_threshold = mp_settings.motion_correlation_min_cod
        self.gripper_depth = gripper_depth
        self.focal_length = focal_length
    
    def process_sequence(
            self, seq_folder, seq_index, add_target_sample=False):
        self.mp.load_buffer_from_disk(seq_folder)
        if add_target_sample:
            self.mp.add_sample(
                self.target_rgb_image, self.target_arm_mask,
                self.target_ee_point, self.target_action_index)
        if self.algorithm == "correlation":
            gripper_depth = None
            focal_length = None
            if self.mp.gripper_depth is None:
                gripper_depth = self.gripper_depth
            if self.mp.focal_length is None:
                focal_length = self.focal_length
            t_0 = time.time()
            self.mp.compute_correlation_images(
                gripper_depth, focal_length)
            t_1 = time.time()
            print(
                f"    compute_correlation_images took {t_1-t_0:.4f} s. "
                f"buffer_size is {self.mp.buffer_size()} samples")
        elif self.algorithm == "segmentation":
            t_0 = time.time()
            self.mp.compute_segmentation_images()
            t_1 = time.time()
            print(
                f"    compute_segmentation_images took {t_1-t_0:.4f} s. "
                f"buffer_size is {self.mp.buffer_size()} samples")
        else:
            raise ValueError(
                f"Unexpected motion perception algorithm: {self.algorithm}.")
        grasped_cable_path = os.path.join(
            seq_folder, "grasped_cable_00000000.png")
        grasped_copy_path = os.path.join(
            self.folder_out, f"grasped_{seq_index}.png")
        shutil.copy2(grasped_cable_path, grasped_copy_path)
        for action, action_idx in self.mp.action2index.items():
            corr_image = self.mp.corr_imgs[action_idx]
            corr_path = os.path.join(
                self.folder_out, f"corr_{seq_index}_{action_idx}.png")
            if self.algorithm == "correlation":
                correlation_mask = corr_image > self.corr_threshold
                cv2.imwrite(corr_path, 255*correlation_mask.astype(np.uint8))
                cod_image = deepcopy(self.mp.cod_images[action_idx])
                a_image = deepcopy(self.mp.a_images[action_idx])
                flow_std_image = deepcopy(self.mp.flow_std_images[action_idx])
                flow_std_image = image_to_uint8(
                    flow_std_image, max_value=0.024, multiplier=1e4)
                cod_image[np.isnan(cod_image)] = 0
                a_image[np.isnan(a_image)] = 0
                abs_a_image = np.abs(a_image)
                abs_a_image = image_to_uint8(abs_a_image)
                cod_image = image_to_uint8(cod_image)
                cod_path = os.path.join(
                    self.folder_out, f"cod_{seq_index}_{action_idx}.png")
                abs_a_path = os.path.join(
                    self.folder_out, f"abs_a_{seq_index}_{action_idx}.png")
                flow_std_path = os.path.join(
                    self.folder_out, f"flow_std_{seq_index}_{action_idx}.png")
                cv2.imwrite(cod_path, cod_image)
                cv2.imwrite(abs_a_path, abs_a_image)
                cv2.imwrite(flow_std_path, flow_std_image)
            elif self.algorithm == "segmentation":
                corr_image[corr_image > 255] = 255
                cv2.imwrite(corr_path, corr_image.astype(np.uint8))
            else:
                raise ValueError(
                    "Unexpected motion perception algorithm: {self.algorithm}.")
        
    def main(self):
        pathlib.Path(self.folder_out).mkdir(parents=True, exist_ok=True)
        seq_index = len(self.sequence_folders) - 1
        seq_folder = self.sequence_folders[seq_index]
        print(f"Processing {seq_folder}...")
        self.process_sequence(seq_folder, seq_index)
        self.target_rgb_image = deepcopy(
            self.mp.image_buffer[-1]["rgb_image"])
        if "arm_mask" in self.mp.image_buffer[-1]:
            self.target_arm_mask = deepcopy(
                self.mp.image_buffer[-1]["arm_mask"])
        else:
            self.target_arm_mask = None
        self.target_ee_point = deepcopy(self.mp.ee_point_buffer[-1])
        self.target_action_index = self.mp.action_buffer[-1]
        target_rgb_path = os.path.join(self.folder_out, "target_rgb.png")
        cv2.imwrite(target_rgb_path, self.target_rgb_image[...,::-1])
        for seq_index, seq_folder in enumerate(self.sequence_folders[0:-1]):
            print(f"Processing {seq_folder}...")
            self.process_sequence(
                seq_folder, seq_index, add_target_sample=True)


def draw_mask(rgb_image, mask, colormap_color, mask_opacity):
    color = np.array(list(colormap_color[0:3]))*255
    rgb_image[mask,:] = (
        (1-mask_opacity)*rgb_image[mask,:]
        + mask_opacity*color[None, None,:])


def draw_segmentation_image(
        target_rgb, sequence_codes, results_folder, color_mode,
        mask_threshold, max_votes=8):
    assert(color_mode in ["multicolor", "voting"])
    mask_opacity = 0.7
    mask_colors = plt.get_cmap('tab20c')
    color_index = 0
    vote_image = np.zeros((target_rgb.shape[0], target_rgb.shape[1]))
    for seq_idx, actions in sorted(sequence_codes.items()):
        if color_mode == "multicolor":
            grasped_mask_path = os.path.join(
                results_folder, f"grasped_{seq_idx}.png")
            grasped_mask = cv2.imread(grasped_mask_path, cv2.IMREAD_UNCHANGED)
            draw_mask(
                target_rgb, grasped_mask>0,
                mask_colors(color_index), mask_opacity)
            color_index += 1
        for action_idx in sorted(actions):
            corr_path = os.path.join(
                results_folder, f"corr_{seq_idx}_{action_idx}.png")
            corr_mask = cv2.imread(corr_path, cv2.IMREAD_UNCHANGED)
            if color_mode == "multicolor":
                draw_mask(
                    target_rgb, corr_mask>mask_threshold,
                    mask_colors(color_index), mask_opacity)
                color_index += 1
            elif color_mode == "voting":
                vote_image[corr_mask>mask_threshold] += 1
            else:
                raise ValueError(f"Unexpected color_mode: {color_mode}")
        while color_index%4 != 0:
            color_index += 1
    if color_mode == "voting":
        vote_image[vote_image>max_votes] = max_votes
        target_rgb[vote_image>0,1] = vote_image[vote_image>0]*(255.0/max_votes)
        target_rgb[vote_image>0,0] = 0
        target_rgb[vote_image>0,2] = 0


def load_sequence_action_codes(results_folder):
    sequence_codes = {}
    for name in os.listdir(results_folder):
        if not name.startswith("corr_"):
            continue
        sp_1 = name.split('_')
        if len(sp_1) != 3:
            continue
        sp_2 = sp_1[2].split('.')
        if len(sp_2) != 2:
            continue
        seq_idx = int(sp_1[1])
        action_idx = int(sp_2[0])
        if seq_idx in sequence_codes:
            sequence_codes[seq_idx].add(action_idx)
        else:
            sequence_codes[seq_idx] = set()
            sequence_codes[seq_idx].add(action_idx)
    return sequence_codes


def view_segmentation(results_folder, algorithm, save_path=None):
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    mask_threshold = 0
    if algorithm == "correlation":
        mask_threshold = 0
    elif algorithm == "segmentation":
        mask_threshold = mp_settings.motion_segmentation_vote_thr
        assert(mask_threshold >= 0)
        assert(mask_threshold <= 100)
    else:
        raise ValueError(
            f"Unexpected motion perception algorithm: {algorithm}.")
    target_rgb_path = os.path.join(results_folder, "target_rgb.png")
    target_rgb = cv2.imread(target_rgb_path, cv2.IMREAD_UNCHANGED)
    assert(target_rgb is not None)
    target_rgb = target_rgb[...,::-1]
    target_rgb = target_rgb.astype(np.float32)
    target_rgb_voting = deepcopy(target_rgb)
    sequence_codes = load_sequence_action_codes(results_folder)
    draw_segmentation_image(
        target_rgb, sequence_codes, results_folder, "multicolor",
        mask_threshold)
    target_rgb = target_rgb.astype(np.uint8)
    draw_segmentation_image(
        target_rgb_voting, sequence_codes, results_folder, "voting",
        mask_threshold)
    target_rgb_voting = target_rgb_voting.astype(np.uint8)
    if save_path is None:
        plt.imshow(target_rgb)
        plt.show()
        plt.imshow(target_rgb_voting)
        plt.show()
    else:
        cv2.imwrite(save_path, target_rgb[...,::-1])
        root, ext = os.path.splitext(save_path)
        save_path_voting = root + "-votes" + ext
        cv2.imwrite(save_path_voting, target_rgb_voting[...,::-1])


def cmd_main():
    """Run the CLI stand-alone program."""
    parser = argparse.ArgumentParser(
        description="Motion perception CLI")
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser_compute = subparsers.add_parser(
        "compute", help=("Run motion perception (correlation) on one or more "
        "image sequence(s). Save the outputs."))
    parser_compute.add_argument(
        "-g", "--gpu", action="store_true", default=False,
        help="run the method on a GPU")
    parser_compute.add_argument(
        "-c", "--crop", default=None, type=int, nargs=4,
        help=("crop the images before computing the flow "
        "(crop origin x, origin y, width, height)"))
    parser_compute.add_argument(
        "-s", "--scale", default=None, type=float,
        help="scale the (cropped) images by this factor before computing flow")
    parser_compute.add_argument(
        "algorithm", default=None, type=str,
        choices=MOTION_PERCEPTION_ALGORITHMS,
        help="motion perception algorithm")
    parser_compute.add_argument(
        "folder_out", default=None, type=str,
        help="output folder path")
    parser_compute.add_argument(
        "sequence_folders", default=None, type=str,
        nargs='+', help="input sequence folder(s)")
    parser_view = subparsers.add_parser(
        "view", help="Show motion perception (correlation) results.")
    parser_view.add_argument(
        "algorithm", default=None, type=str,
        choices=MOTION_PERCEPTION_ALGORITHMS,
        help="motion perception algorithm")
    parser_view.add_argument(
        "results_folder", default=None, type=str,
        help="folder containing the motion perception results")
    args = parser.parse_args()
    if args.subparser_name == "compute":
        crop_origin = None
        crop_size = None
        scale_factor = None
        if args.crop is not None:
            crop_origin = args.crop[0:2]
            crop_size = args.crop[2:]
        if args.scale is not None:
            scale_factor = args.scale
        mpm = MotionPerceptionMulti(
            args.folder_out, args.sequence_folders, args.gpu, args.algorithm,
            crop_origin=crop_origin, crop_size=crop_size,
            scale_factor=scale_factor)
        mpm.main()
    elif args.subparser_name == "view":
        view_segmentation(args.algorithm, args.results_folder)
    else:
        raise ValueError(
            f"Unexpected command (subparser_name): {args.subparser_name}")


if __name__ == "__main__":
    cmd_main()
