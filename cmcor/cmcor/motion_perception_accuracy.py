"""
Compute and print the accuracy of motion correlation and segmentation.
This requires having their pre-computed results saved on the disk.
"""

import os
import json
import pathlib
import argparse
import pprint
from copy import deepcopy

import cv2
import numpy as np
import matplotlib
# https://phyletica.org/matplotlib-fonts/ - use TrueType fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from cmcor import motion_perception_cli
from cmcor import motion_perception_multi_eval
from cmcor import motion_perception_settings



def load_segmentation_predictions(
        results_path, sequence_codes, param, image_prefix="corr_"):
    masks = []
    for seq_idx, actions in sorted(sequence_codes.items()):
        for action_idx in sorted(actions):
            corr_path = os.path.join(
                results_path, f"{image_prefix}{seq_idx}_{action_idx}.png")
            corr_image = cv2.imread(corr_path, cv2.IMREAD_UNCHANGED)
            masks.append(corr_image > param)
    return masks


def load_correlation_predictions(
        results_path, sequence_codes, param):
    masks = []
    for seq_idx, actions in sorted(sequence_codes.items()):
        for action_idx in sorted(actions):
            cod_path = os.path.join(
                results_path, f"cod_{seq_idx}_{action_idx}.png")
            abs_a_path = os.path.join(
                results_path, f"abs_a_{seq_idx}_{action_idx}.png")
            cod_image = cv2.imread(cod_path, cv2.IMREAD_UNCHANGED)
            abs_a_image = cv2.imread(abs_a_path, cv2.IMREAD_UNCHANGED)
            min_cod, min_abs_a = param
            mask = np.logical_and(
                abs_a_image >= min_abs_a, cod_image >= min_cod)
            masks.append(mask)
    return masks


def load_predictions(results_path, sequence_codes, param, method_title):
    if method_title == "Motion correlation":
        masks = load_correlation_predictions(
            results_path, sequence_codes, param)
    elif method_title == "Motion segmentation":
        masks = load_segmentation_predictions(
            results_path, sequence_codes, param, "corr_")
    elif method_title == "Flow standard deviation":
        masks = load_segmentation_predictions(
            results_path, sequence_codes, param, "flow_std_")
    else:
        raise ValueError(f"Unexpected method title (name): {method_title}")
    vote_image = None
    for corr_mask in masks:
        if vote_image is None:
            vote_image = corr_mask.astype(int)
        else:
            vote_image[corr_mask] += 1
    return vote_image


def load_predictions_multi_grasp(
        results_path, sequence_codes, param, method_title):
    if method_title == "Motion correlation":
        masks = load_correlation_predictions(
            results_path, sequence_codes, param)
    elif method_title == "Motion segmentation":
        masks = load_segmentation_predictions(
            results_path, sequence_codes, param)
    else:
        raise ValueError(f"Unexpected method title (name): {method_title}")

    vote_images = []
    mask_idx = 0
    for seq_idx, actions in sorted(sequence_codes.items()):
        tmp_vote_image = None
        for action_idx in sorted(actions):
            corr_mask = masks[mask_idx]
            mask_idx += 1
            if tmp_vote_image is None:
                tmp_vote_image = corr_mask.astype(int)
            else:
                tmp_vote_image[corr_mask] += 1
        vote_image = deepcopy(tmp_vote_image)
        if vote_images:
            vote_image += vote_images[-1]
        vote_images.append(vote_image)
    return vote_images


def get_f_beta(beta, true_positive, false_negative, false_positive):
    f_beta = (1+beta**2)*true_positive
    den = ((1+beta**2)*true_positive + (beta**2)*false_negative
           + false_positive)
    if den > 0:
        f_beta = f_beta/den
    else:
        f_beta = np.nan
    return f_beta


def get_mask_metrics(pred_mask, gt_mask, beta=1):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    n_gt = float(np.sum(gt_mask))
    n_pred = float(np.sum(pred_mask))
    n_intersection = float(np.sum(intersection))
    n_union = float(np.sum(union))
    iou = np.nan
    recall = np.nan
    precision = np.nan
    if n_gt > 0:
        recall = n_intersection/n_gt
    if n_pred > 0:
        precision = n_intersection/n_pred
    if n_union > 0:
        iou = n_intersection/n_union
    false_positive = n_pred - n_intersection
    false_negative = n_gt - n_intersection
    f_beta = get_f_beta(beta, n_intersection, false_negative, false_positive)
    return iou, recall, precision, f_beta



def print_mean_metrics(results_dict, method_title):
    print("Method & Recall & Precision & F_beta & IoU\\")
    for vote in results_dict["IoU"]:
        mean_iou = np.nanmean(results_dict["IoU"][vote])
        mean_recall = np.nanmean(results_dict["recall"][vote])
        mean_precision = np.nanmean(results_dict["precision"][vote])
        mean_f_beta = np.nanmean(results_dict["F_beta"][vote])
        print(
            f"{method_title} @ {vote} & {mean_recall:.4f} & "
            f"{mean_precision:.4f} & {mean_f_beta:.4f} & {mean_iou:.4f}\\\\")



def compute_accuracy(
        folder_results, folder_gt, method_title, data_root=None,
        beta=1.0, param=None, max_votes=2, verbose=False):
    """
    IoU/recall/precision -> vote threshold -> recording
    """
    assert(param is not None)
    vote_thresholds = [i for i in range(1, max_votes+1)]
    results_dict = {
        "IoU": {v:[] for v in vote_thresholds},
        "recall": {v:[] for v in vote_thresholds},
        "precision": {v:[] for v in vote_thresholds},
        "F_beta": {v:[] for v in vote_thresholds}}
    print(f"======== {method_title}, beta = {beta} ========")
    for name in sorted(os.listdir(folder_gt)):
        if not name.startswith("20"):
            continue
        results_path = os.path.join(folder_results, name)
        if not os.path.isdir(results_path):
            continue
        gt_path = os.path.join(folder_gt, name)
        gt_images = sorted(os.listdir(gt_path))
        if not gt_images:
            continue
        gt_img_path = os.path.join(gt_path, gt_images[0])
        gt_image = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
        gt_mask = gt_image > 0
        sequence_codes = motion_perception_cli.load_sequence_action_codes(
            results_path)
        vote_image = load_predictions(
            results_path, sequence_codes, param, method_title)
        if data_root is not None:
            json_path = os.path.join(data_root, name, "actions_gripper.json")
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
            if "crop_origin" in json_dict and "crop_size" in json_dict:
                crop_origin = json_dict["crop_origin"]
                crop_size = json_dict["crop_size"]
                vote_image = vote_image[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
                gt_mask = gt_mask[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
        for thr in vote_thresholds:
            mask = vote_image >= thr
            metrics = get_mask_metrics(mask, gt_mask, beta)
            iou, recall, precision, f_beta = metrics
            results_dict["IoU"][thr].append(iou)
            results_dict["recall"][thr].append(recall)
            results_dict["precision"][thr].append(precision)
            results_dict["F_beta"][thr].append(f_beta)
    if verbose:
        pprint.pprint(results_dict)
    print_mean_metrics(results_dict, method_title)


def get_multigrasp_eval_data(
        seq, param, method_title, folder_results, folder_gt, data_root):
    multi_name = motion_perception_multi_eval.get_multi_name(seq)
    results_path = os.path.join(folder_results, multi_name)
    if not os.path.isdir(results_path):
        return None, None
    gt_path = os.path.join(folder_gt, seq[-1])
    if not os.path.exists(gt_path):
        return None, None
    gt_images = sorted(os.listdir(gt_path))
    if not gt_images:
        return None, None
    gt_img_path = os.path.join(gt_path, gt_images[0])
    gt_image = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
    gt_mask = gt_image > 0
    sequence_codes = motion_perception_cli.load_sequence_action_codes(
        results_path)
    vote_images = load_predictions_multi_grasp(
        results_path, sequence_codes, param, method_title)
    if data_root is not None:
        json_path = os.path.join(
            data_root, seq[-1], "actions_gripper.json")
        with open(json_path, "r") as fp:
            json_dict = json.load(fp)
        if "crop_origin" in json_dict and "crop_size" in json_dict:
            crop_origin = json_dict["crop_origin"]
            crop_size = json_dict["crop_size"]
            for image_idx in range(len(vote_images)):
                vote_images[image_idx] = vote_images[image_idx][
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
            gt_mask = gt_mask[
                crop_origin[1]:crop_origin[1]+crop_size[1],
                crop_origin[0]:crop_origin[0]+crop_size[0]]
    return vote_images, gt_mask


def compute_accuracy_multi_grasp(
        sequence_names, folder_results, folder_gt, method_title,
        data_root=None,
        beta=1.0, parameters=None, verbose=False):
    """
    IoU/recall/precision -> vote threshold -> recording
    """
    max_votes_double=4
    max_votes_single=2
    assert(parameters is not None)
    vote_thresholds_double = [i for i in range(1, max_votes_double+1)]
    results_dict = {
        "IoU": {v:[] for v in vote_thresholds_double},
        "recall": {v:[] for v in vote_thresholds_double},
        "precision": {v:[] for v in vote_thresholds_double},
        "F_beta": {v:[] for v in vote_thresholds_double}}
    vote_thresholds_single = [i for i in range(1, max_votes_single+1)]
    results_dict_single = {
        "IoU": {v:[] for v in vote_thresholds_single},
        "recall": {v:[] for v in vote_thresholds_single},
        "precision": {v:[] for v in vote_thresholds_single},
        "F_beta": {v:[] for v in vote_thresholds_single}}
    print(f"======== {method_title}, beta = {beta} ========")
    for seq, param in zip(sequence_names, parameters):
        vote_images, gt_mask = get_multigrasp_eval_data(
            seq, param, method_title, folder_results, folder_gt, data_root)
        if vote_images is None or gt_mask is None:
            continue
        for thr in vote_thresholds_double:
            mask = vote_images[-1] >= thr
            metrics = get_mask_metrics(mask, gt_mask, beta)
            iou, recall, precision, f_beta = metrics
            results_dict["IoU"][thr].append(iou)
            results_dict["recall"][thr].append(recall)
            results_dict["precision"][thr].append(precision)
            results_dict["F_beta"][thr].append(f_beta)
        for thr in vote_thresholds_single:
            mask_single = vote_images[0] >= thr
            metrics = get_mask_metrics(mask_single, gt_mask, beta)
            iou, recall, precision, f_beta = metrics
            results_dict_single["IoU"][thr].append(iou)
            results_dict_single["recall"][thr].append(recall)
            results_dict_single["precision"][thr].append(precision)
            results_dict_single["F_beta"][thr].append(f_beta)
    if verbose:
        print("One grasp:")
        pprint.pprint(results_dict_single)
        print("All grasps:")
        pprint.pprint(results_dict)
    print("One grasp:")
    print_mean_metrics(results_dict_single, method_title)
    print("All grasps:")
    print_mean_metrics(results_dict, method_title)


def grasps_vs_accuracy(
        sequence_names, folder_results, folder_gt, method_title,
        data_root=None,
        beta=1.0, parameters=None, plot_means=True):
    vote_threshold = 2
    assert(parameters is not None)
    # indexing example: results_dict["F_beta"][seq_idx][grasp_no]
    results_dict = {
        "IoU": [[0] for seq in sequence_names],
        "recall": [[0] for seq in sequence_names],
        "precision": [[0] for seq in sequence_names],
        "F_beta": [[0] for seq in sequence_names],
        }
    print(f"======== {method_title}, beta = {beta} ========")
    for seq_idx, (seq, param) in enumerate(zip(sequence_names, parameters)):
        vote_images, gt_mask = get_multigrasp_eval_data(
            seq, param, method_title, folder_results, folder_gt, data_root)
        if vote_images is None:
            continue
        for grasp_idx in range(len(vote_images)):
            mask = vote_images[grasp_idx] >= vote_threshold
            metrics = get_mask_metrics(mask, gt_mask, beta)
            iou, recall, precision, f_beta = metrics
            results_dict["IoU"][seq_idx].append(iou)
            results_dict["recall"][seq_idx].append(recall)
            results_dict["precision"][seq_idx].append(precision)
            results_dict["F_beta"][seq_idx].append(f_beta)
    plt.rcParams['font.family'] = "DejaVu Sans" #"FreeSerif" #"Times New Roman"
    plt.rcParams['font.size'] = 16
    linewidth = 3
    if plot_means:
        max_grasps = 6+1
        grasps_no_cut = 5+1
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5.25,4.28)
        recalls = np.zeros((len(sequence_names), max_grasps))
        recalls.fill(np.nan)
        precisions = np.zeros((len(sequence_names), max_grasps))
        precisions.fill(np.nan)
        f_betas = np.zeros((len(sequence_names), max_grasps))
        f_betas.fill(np.nan)
        for seq_idx in range(len(sequence_names)):
            n_grasps = len(results_dict["recall"][seq_idx])
            #grasp_numbers = np.arange(len(results_dict["recall"][seq_idx]))
            recalls[seq_idx, 0:n_grasps] = results_dict["recall"][seq_idx]
            precisions[seq_idx, 0:n_grasps] = results_dict["precision"][seq_idx]
            f_betas[seq_idx, 0:n_grasps] = results_dict["F_beta"][seq_idx]
        grasp_numbers = np.arange(max_grasps)
        mean_recalls = np.nanmean(recalls, axis=0)
        mean_precisions = np.nanmean(precisions, axis=0)
        mean_f_betas = np.nanmean(f_betas, axis=0)
        grasp_numbers = grasp_numbers[0:grasps_no_cut]
        mean_recalls = mean_recalls[0:grasps_no_cut]
        mean_precisions = mean_precisions[0:grasps_no_cut]
        mean_f_betas = mean_f_betas[0:grasps_no_cut]
        ax.plot(
            grasp_numbers, mean_precisions, label="Precision",
            linestyle="-", linewidth=linewidth, marker='+')
        ax.plot(
            grasp_numbers, mean_f_betas, label=f"F_{beta}",
            linestyle="--", linewidth=linewidth, marker='D')
        ax.plot(
            grasp_numbers, mean_recalls, label="Recall",
            linestyle=":", linewidth=linewidth, marker='o')
        ax.set_xlabel("# grasps")
        ax.set_ylabel(f"Recall / Precision / F_{beta}")
        ax.set_ylim(0,1)
        ax.set_title("Multiple Grasps")
        plt.legend()
    else:
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(10.5,4.28)
        for seq_idx in range(len(sequence_names)):
            grasp_numbers = np.arange(len(results_dict["recall"][seq_idx]))
            recall = results_dict["recall"][seq_idx]
            precision = results_dict["precision"][seq_idx]
            f_beta = results_dict["F_beta"][seq_idx]
            axs[0].plot(grasp_numbers, recall)
            axs[1].plot(grasp_numbers, precision)
            axs[2].plot(grasp_numbers, f_beta)
        #axs[0].set_title("Recall")
        axs[0].set_xlabel("# grasps")
        axs[0].set_ylabel("Recall")
        #axs[1].set_title("Precision")
        axs[1].set_xlabel("# grasps")
        axs[1].set_ylabel("Precision")
        axs[2].set_xlabel("# grasps")
        axs[2].set_ylabel(f"F_{beta}")
    plt.tight_layout()
    plt.savefig(f"grasps-vs-accuracy-vote-thr-{vote_threshold}.pdf")


def single_grasp_evaluation():
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    gt_root = mp_settings.gt_root
    out_root = mp_settings.output_root
    data_root = mp_settings.data_root
    folder_out_correlation = os.path.join(out_root, "correlation_masks")
    folder_out_segmentation = os.path.join(out_root, "segmentation_masks")
    gt_root_validation = os.path.join(gt_root, "validation")
    gt_root_test = os.path.join(gt_root, "test")
    all_betas = mp_settings.eval_all_betas
    params_seg = mp_settings.eval_params_segmentation
    params_corr = mp_settings.eval_params_correlation
    params_flow_std = mp_settings.eval_params_flow_std
    print("==================== Single grasp evaluation ====================")
    print("================ Validation set ================")
    for beta, p_seg, p_corr, p_std in zip(
            all_betas, params_seg, params_corr, params_flow_std):
        compute_accuracy(
            folder_out_correlation, gt_root_validation, "Motion correlation",
            data_root, beta, p_corr)
        compute_accuracy(
            folder_out_correlation, gt_root_validation,
            "Flow standard deviation", data_root, beta, p_std)
        compute_accuracy(
            folder_out_segmentation, gt_root_validation, "Motion segmentation",
            data_root, beta, p_seg)
    print("")
    print("================ Test set ================")
    for beta, p_seg, p_corr, p_std in zip(
            all_betas, params_seg, params_corr, params_flow_std):
        compute_accuracy(
            folder_out_correlation, gt_root_test, "Motion correlation",
            data_root, beta, p_corr)
        compute_accuracy(
            folder_out_correlation, gt_root_test,
            "Flow standard deviation", data_root, beta, p_std)
        compute_accuracy(
            folder_out_segmentation, gt_root_test, "Motion segmentation",
            data_root, beta, p_seg)


def double_grasp_evaluation():
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    gt_root = mp_settings.gt_root
    out_root = mp_settings.output_root
    data_root = mp_settings.data_root
    folder_out_correlation = os.path.join(out_root, "correlation_masks_multi")
    gt_root_test = os.path.join(gt_root, "test")
    sequences = mp_settings.double_grasp_sequences
    params = mp_settings.double_grasp_params
    print("==================== Double grasp evaluation ====================")
    grasps_vs_accuracy(
        sequences, folder_out_correlation, gt_root_test,
        "Motion correlation", data_root, beta=0.4,
        parameters=params)
    compute_accuracy_multi_grasp(
        sequences, folder_out_correlation, gt_root_test, "Motion correlation",
        data_root, beta=0.4, parameters=params, verbose=False)



if __name__ == "__main__":
    single_grasp_evaluation()
    double_grasp_evaluation()
