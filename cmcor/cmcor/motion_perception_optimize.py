"""
Find the optimal motion correlation and segmentation method parameters.
Use only the validation set.
This requires having pre-computed (validation) results saved on the disk.
"""

import os
import json
import pathlib
import argparse
import pprint

import cv2
import numpy as np

from cmcor import motion_perception_cli
from cmcor.motion_perception_accuracy import get_f_beta
from cmcor import motion_perception_settings



def load_segmentation_predictions(
        results_path, sequence_codes, image_prefix="corr_",
        crop_origin=None, crop_size=None):
    corr_images = []
    for seq_idx, actions in sorted(sequence_codes.items()):
        seq_pred = []
        for action_idx in sorted(actions):
            corr_path = os.path.join(
                results_path, f"{image_prefix}{seq_idx}_{action_idx}.png")
            corr_image = cv2.imread(corr_path, cv2.IMREAD_UNCHANGED)
            if crop_origin is not None and crop_size is not None:
                corr_image = corr_image[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
            seq_pred.append(corr_image)
        corr_images.append(seq_pred)
    return corr_images


def load_correlation_predictions(
        results_path, sequence_codes, crop_origin=None, crop_size=None):
    cod_a_images = []
    for seq_idx, actions in sorted(sequence_codes.items()):
        seq_pred = []
        for action_idx in sorted(actions):
            cod_path = os.path.join(
                results_path, f"cod_{seq_idx}_{action_idx}.png")
            abs_a_path = os.path.join(
                results_path, f"abs_a_{seq_idx}_{action_idx}.png")
            cod_image = cv2.imread(cod_path, cv2.IMREAD_UNCHANGED)
            abs_a_image = cv2.imread(abs_a_path, cv2.IMREAD_UNCHANGED)
            if crop_origin is not None and crop_size is not None:
                cod_image = cod_image[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
                abs_a_image = abs_a_image[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
            seq_pred.append({"cod":cod_image, "abs_a":abs_a_image})
        cod_a_images.append(seq_pred)
    return cod_a_images


def load_data(folder_results, folder_gt, method_title, data_root):
    predictions = []
    gt_masks = []
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

        crop_size = None
        crop_origin = None
        if data_root is not None:
            json_path = os.path.join(data_root, name, "actions_gripper.json")
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
            if "crop_origin" in json_dict and "crop_size" in json_dict:
                crop_origin = json_dict["crop_origin"]
                crop_size = json_dict["crop_size"]
                gt_mask = gt_mask[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]

        if method_title == "Motion correlation":
            pred = load_correlation_predictions(
                results_path, sequence_codes, crop_origin, crop_size)
        elif method_title == "Motion segmentation":
            pred = load_segmentation_predictions(
                results_path, sequence_codes, "corr_", crop_origin, crop_size)
        elif method_title == "Flow standard deviation":
            pred = load_segmentation_predictions(
                results_path, sequence_codes, "flow_std_",
                crop_origin, crop_size)
        else:
            raise ValueError(f"Unexpected method title (name): {method_title}")
        predictions.extend(pred)
        gt_masks.extend(len(pred)*[gt_mask])
    assert(len(predictions) == len(gt_masks))
    return predictions, gt_masks


def report_best(metric_names, metric_values, parameter_space):
    for metric_idx, name in enumerate(metric_names):
        best_idx = np.argmax(metric_values[:, metric_idx])
        best_param = parameter_space[best_idx]
        print(f"The best {name} is for parameter {best_param}. Its metrics are:")
        for metric_idx_2, name_2 in enumerate(metric_names):
            val = metric_values[best_idx, metric_idx_2]
            print(f"    {name_2} {val:.4f}")


def get_mask_metrics(pred_mask, gt_mask):
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
    f1 = get_f_beta(1, n_intersection, false_negative, false_positive)
    f05 = get_f_beta(0.5, n_intersection, false_negative, false_positive)
    f04 = get_f_beta(0.4, n_intersection, false_negative, false_positive)
    f035 = get_f_beta(0.35, n_intersection, false_negative, false_positive)
    f03 = get_f_beta(0.3, n_intersection, false_negative, false_positive)
    f02 = get_f_beta(0.2, n_intersection, false_negative, false_positive)
    f01 = get_f_beta(0.1, n_intersection, false_negative, false_positive)
    return iou, recall, precision, f1, f05, f04, f035, f03, f02, f01


def get_metric_names():
    return [
            "IoU", "Recall", "Precision",
            "F1", "F0.5", "F0.4", "F0.35", "F0.3", "F0.2", "F0.1"]


def optimize_parameters(
        folder_results, folder_gt, method_title, data_root=None):
    """
    IoU/recall/precision -> vote threshold -> recording
    """
    print(f"======== {method_title} ========")
    predictions, gt_masks = load_data(
        folder_results, folder_gt, method_title, data_root)
    if method_title == "Motion correlation":
        parameter_space = []
        for cod in range(1,100):
            for abs_a in range(1,100):
                parameter_space.append([cod, abs_a])
    elif method_title == "Motion segmentation":
        parameter_space = [thr for thr in range(1,100)]
    elif method_title == "Flow standard deviation":
        parameter_space = [thr for thr in range(1,240)]
    else:
        raise ValueError(f"Unexpected method title (name): {method_title}")
    metric_names = get_metric_names()
    metric_values = np.zeros((len(parameter_space), len(metric_names)))
    for idx, param in enumerate(parameter_space):
        tmp_metrics = []
        for action_preds, gt_mask in zip(predictions, gt_masks):
            mask = None
            for pred in action_preds:
                if method_title == "Motion correlation":
                    min_cod, min_abs_a = param
                    tmp_mask = np.logical_and(
                        pred["abs_a"] >= min_abs_a, pred["cod"] >= min_cod)
                elif method_title == "Motion segmentation":
                    tmp_mask = pred > param
                elif method_title == "Flow standard deviation":
                    tmp_mask = pred > param
                else:
                    raise ValueError(f"Unexpected method title (name): {method_title}")
                if mask is None:
                    mask = tmp_mask
                else:
                    mask = np.logical_or(mask, tmp_mask)
            tmp_metrics.append(get_mask_metrics(mask, gt_mask))
        tmp_metrics = np.array(tmp_metrics)
        metric_values[idx,:] = np.nanmean(tmp_metrics, axis=0)
    metric_values[np.isnan(metric_values)] = 0
    report_best(metric_names, metric_values, parameter_space)


if __name__ == "__main__":
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    gt_root = mp_settings.gt_root
    out_root = mp_settings.output_root
    data_root = mp_settings.data_root
    folder_out_correlation = os.path.join(out_root, "correlation_masks")
    folder_out_segmentation = os.path.join(out_root, "segmentation_masks")
    gt_root_validation = os.path.join(gt_root, "validation")
    print("================ Validation set ================")
    optimize_parameters(
        folder_out_segmentation, gt_root_validation, "Motion segmentation",
        data_root)
    optimize_parameters(
        folder_out_correlation, gt_root_validation, "Flow standard deviation",
        data_root)
    optimize_parameters(
        folder_out_correlation, gt_root_validation, "Motion correlation",
        data_root)

