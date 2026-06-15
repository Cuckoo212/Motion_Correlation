#!/usr/bin/env python3
"""Recompute an orchestrator-style next grasp plan from saved CMCor masks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CABLE_INTERACT_ROOT = Path(
    "/home/flexcycle/franka_ros2_ws/src/cable_interact"
)
DEFAULT_HANDEYE_CALIBRATION_PATH = Path(
    "/home/flexcycle/.ros2/easy_handeye2/calibrations/fr3_calibration.calib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the files in a grasp_sample_XXX directory from saved "
            "CMCor correlation masks without starting ROS or the controller."
        )
    )
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("correlation_dir", type=Path)
    parser.add_argument("previous_grasp_plan", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--cable-interact-root",
        type=Path,
        default=DEFAULT_CABLE_INTERACT_ROOT,
    )
    parser.add_argument(
        "--handeye-calibration-path",
        type=Path,
        default=DEFAULT_HANDEYE_CALIBRATION_PATH,
    )
    parser.add_argument(
        "--gripper-direction-sample-count",
        type=int,
        default=36,
    )
    return parser.parse_args()


def add_import_path(path: Path) -> None:
    path_text = str(path.expanduser().resolve())
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_last_frame(sequence_dir: Path, prefix: str) -> tuple[Path, np.ndarray]:
    paths = sorted(sequence_dir.glob(f"{prefix}_*.png"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_*.png files found in {sequence_dir}")
    flag = cv2.IMREAD_COLOR if prefix == "rgb" else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(paths[-1]), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {paths[-1]}")
    return paths[-1], image


def load_motion_masks(correlation_dir: Path) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(correlation_dir.glob("corr_*.png"))
    if not paths:
        raise FileNotFoundError(f"No corr_*.png files found in {correlation_dir}")
    result = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read correlation mask: {path}")
        result.append((path, image > 0))
    return result


def build_vote_image(motion_masks: list[tuple[Path, np.ndarray]]) -> np.ndarray:
    vote_image = np.zeros(motion_masks[0][1].shape, dtype=np.float32)
    for _, mask in motion_masks:
        vote_image[mask] += 1.0
    return vote_image


def transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    if points.size == 0:
        return points
    return (points @ rotation.T) + translation[None, :]


def robot_point_to_camera(
    point_robot: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return rotation.T @ (point_robot - translation)


def build_gripper_directions(axis: np.ndarray, sample_count: int) -> list[list[float]]:
    axis = axis.astype(np.float64)
    axis /= np.linalg.norm(axis)
    reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(axis.dot(reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    basis_v /= np.linalg.norm(basis_v)
    directions = []
    for idx in range(sample_count):
        theta = 2.0 * math.pi * idx / sample_count
        direction = math.cos(theta) * basis_u + math.sin(theta) * basis_v
        direction /= np.linalg.norm(direction)
        directions.append(direction.tolist())
    return directions


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir.expanduser().resolve()
    correlation_dir = args.correlation_dir.expanduser().resolve()
    previous_plan_path = args.previous_grasp_plan.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cable_interact_root = args.cable_interact_root.expanduser().resolve()
    calibration_path = args.handeye_calibration_path.expanduser().resolve()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    add_import_path(cable_interact_root)
    from pointcloud_tools import build_cable_point_cloud
    from cmcor import grasp_segment_sampler

    metadata = load_json(sequence_dir / "actions_gripper.json")
    if "camera_matrix" not in metadata:
        raise RuntimeError("Recorded sequence has no camera_matrix in actions_gripper.json.")
    camera_matrix = np.asarray(metadata["camera_matrix"], dtype=np.float64)
    _, depth = load_last_frame(sequence_dir, "depth")
    _, rgb_bgr = load_last_frame(sequence_dir, "rgb")
    motion_masks = load_motion_masks(correlation_dir)
    vote_image = build_vote_image(motion_masks)
    if not np.any(vote_image > 0):
        raise RuntimeError("Correlation masks contain no foreground pixels.")

    handeye = build_cable_point_cloud.load_handeye_transform(calibration_path)
    rotation = np.asarray(handeye["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(handeye["translation_vector"], dtype=np.float64)
    previous_plan = load_json(previous_plan_path)
    previous_grasp_robot = np.asarray(previous_plan["grasp_point"], dtype=np.float64)
    previous_grasp_camera = robot_point_to_camera(
        previous_grasp_robot, rotation, translation
    )

    sampler = grasp_segment_sampler.GraspSegmentSampler()
    sampler.clear_previous_grasps()
    sampler.add_previous_grasp(previous_grasp_camera)
    sampler.set_input_images(depth, camera_matrix, vote_image)
    sampled = sampler.sample_grasp()
    if sampled is None:
        raise RuntimeError("No valid next grasp was sampled.")

    grasp, grasp_mask = sampled
    center_camera, axis_camera, points_camera = grasp
    center_camera = np.asarray(center_camera, dtype=np.float64)
    axis_camera = np.asarray(axis_camera, dtype=np.float64)
    axis_camera /= np.linalg.norm(axis_camera)
    points_camera = np.asarray(points_camera, dtype=np.float64)
    center_robot = transform_points(center_camera[None, :], rotation, translation)[0]
    axis_robot = transform_points(axis_camera[None, :], rotation, np.zeros(3))[0]
    axis_robot /= np.linalg.norm(axis_robot)
    points_robot = transform_points(points_camera, rotation, translation)

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug" / sequence_dir.name
    debug_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, mask) in enumerate(motion_masks):
        cv2.imwrite(str(debug_dir / f"motion_mask_{index:02d}.png"), 255 * mask.astype(np.uint8))

    grasp_name = output_dir.name
    grasp_mask_path = output_dir / f"{grasp_name}_mask.png"
    cv2.imwrite(str(grasp_mask_path), 255 * (grasp_mask > 0).astype(np.uint8))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    overlay = build_cable_point_cloud.overlay_mask_on_rgb(rgb, grasp_mask > 0)
    cv2.imwrite(
        str(output_dir / f"{grasp_name}_mask_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    build_cable_point_cloud.save_point_cloud_ply(
        str(output_dir / f"{grasp_name}_camera_segment.ply"), points_camera
    )
    build_cable_point_cloud.save_point_cloud_ply(
        str(output_dir / f"{grasp_name}_robot_segment.ply"), points_robot
    )

    plan = {
        "source": "cmcor.recompute_next_grasp",
        "source_sequence": sequence_dir.name,
        "camera_frame": str(handeye["source_frame"]),
        "robot_frame": str(handeye["target_frame"]),
        "grasp_point": center_robot.tolist(),
        "local_tangent": axis_robot.tolist(),
        "gripper_direction_sample_count": args.gripper_direction_sample_count,
        "gripper_directions": build_gripper_directions(
            axis_robot, args.gripper_direction_sample_count
        ),
        "camera_grasp_point": center_camera.tolist(),
        "camera_local_tangent": axis_camera.tolist(),
        "sampled_segment_point_count": int(points_camera.shape[0]),
    }
    plan_path = output_dir / grasp_name
    with plan_path.open("w", encoding="utf-8") as fp:
        json.dump(plan, fp, sort_keys=True, indent=2)
        fp.write("\n")

    print(f"Saved next grasp plan: {plan_path}")
    print(f"Camera grasp point: {center_camera.tolist()}")
    print(f"Robot grasp point: {center_robot.tolist()}")
    print(f"Sampled segment points: {points_camera.shape[0]}")


if __name__ == "__main__":
    main()
