#!/usr/bin/env python3
"""Generate CMCor arm_DDDDDDDD.png masks from RGB frames with SAM3."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_SAM3_PROJECT = (
    Path.home() / "franka_ros2_ws" / "src" / "cable_interact" / "sam3_project"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use SAM3 to generate arm_DDDDDDDD.png masks for one recorded "
            "CMCor sequence. Existing masks are skipped by default."
        )
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Directory containing rgb_DDDDDDDD.png and actions_gripper.json.",
    )
    parser.add_argument(
        "--sam3-project",
        type=Path,
        default=DEFAULT_SAM3_PROJECT,
        help=f"SAM3 source directory (default: {DEFAULT_SAM3_PROJECT}).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="SAM3 checkpoint path (default: <sam3-project>/checkpoints/sam3.pt).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help=(
            "SAM3 text prompt. Repeat to merge masks from multiple prompts "
            "(default: robot arm)."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum SAM3 candidate score (default: 0.5).",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0,
        help="Discard candidates smaller than this fraction of the image.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=1.0,
        help="Discard candidates larger than this fraction of the image.",
    )
    parser.add_argument(
        "--selection",
        choices=("union", "best"),
        default="union",
        help="Merge all retained masks or use only the highest-score mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate masks that already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many RGB frames. Useful for checking quality.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        help="Optional directory for RGB debug overlays.",
    )
    parser.add_argument(
        "--no-update-metadata",
        action="store_true",
        help="Do not set poor_arm_masks=false after all RGB masks exist.",
    )
    return parser.parse_args()


def import_sam3(sam3_project: Path):
    sam3_project = sam3_project.expanduser().resolve()
    if not (sam3_project / "sam3").is_dir():
        raise FileNotFoundError(f"SAM3 source directory not found: {sam3_project}")
    sys.path.insert(0, str(sam3_project))

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    return torch, Sam3Processor, build_sam3_image_model


def save_png(array: np.ndarray, output_path: Path, mode: str) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    Image.fromarray(array, mode=mode).save(temporary_path, format="PNG")
    os.replace(temporary_path, output_path)


def save_overlay(image: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    overlay = image.astype(np.float32)
    foreground = mask > 0
    overlay[foreground] = overlay[foreground] * 0.55 + np.array(
        [255.0, 0.0, 0.0]
    ) * 0.45
    save_png(overlay.astype(np.uint8), output_path, "RGB")


def select_mask(
    output: dict,
    image_area: int,
    min_area_ratio: float,
    max_area_ratio: float,
    selection: str,
) -> np.ndarray | None:
    candidates: list[tuple[float, np.ndarray]] = []
    for mask, score in zip(output["masks"], output["scores"]):
        mask_array = np.squeeze(mask.detach().cpu().numpy()).astype(bool)
        score_value = float(score.detach().cpu().item())
        area_ratio = float(mask_array.sum()) / image_area
        if min_area_ratio <= area_ratio <= max_area_ratio:
            candidates.append((score_value, mask_array))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    if selection == "best":
        return candidates[0][1]
    return np.logical_or.reduce([item[1] for item in candidates])


def set_arm_masks_available(metadata_path: Path) -> None:
    with metadata_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    metadata["poor_arm_masks"] = False
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, sort_keys=True, indent=4)
        fp.write("\n")
    os.replace(temporary_path, metadata_path)


def main() -> None:
    args = parse_args()
    prompts = args.prompt or ["robot arm"]
    sequence_dir = args.sequence_dir.expanduser().resolve()
    sam3_project = args.sam3_project.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else sam3_project / "checkpoints" / "sam3.pt"
    )
    metadata_path = sequence_dir / "actions_gripper.json"
    rgb_paths = sorted(sequence_dir.glob("rgb_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png"))

    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"CMCor sequence directory not found: {sequence_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"CMCor metadata file not found: {metadata_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    if not rgb_paths:
        raise FileNotFoundError(f"No rgb_DDDDDDDD.png frames found in: {sequence_dir}")
    if not 0.0 <= args.min_area_ratio <= args.max_area_ratio <= 1.0:
        raise ValueError("Area ratios must satisfy 0 <= min <= max <= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    pending_paths = [
        rgb_path
        for rgb_path in rgb_paths
        if args.overwrite or not (sequence_dir / rgb_path.name.replace("rgb_", "arm_")).is_file()
    ]
    if args.limit is not None:
        pending_paths = pending_paths[: args.limit]
    if args.overlay_dir:
        args.overlay_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    if pending_paths:
        torch, Sam3Processor, build_sam3_image_model = import_sam3(sam3_project)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3 arm-mask generation requires a CUDA GPU, but PyTorch "
                "cannot access one. Check the NVIDIA driver and run this "
                "command in an environment with GPU access."
            )
        device = "cuda"
        print(f"Loading SAM3 on {device}: {checkpoint_path}")
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            device=device,
        )
        processor = Sam3Processor(
            model,
            device=device,
            confidence_threshold=args.confidence_threshold,
        )

        for index, rgb_path in enumerate(pending_paths, start=1):
            mask_path = sequence_dir / rgb_path.name.replace("rgb_", "arm_")
            print(f"[{index}/{len(pending_paths)}] Segmenting {rgb_path.name}")
            image = Image.open(rgb_path).convert("RGB")
            image_array = np.asarray(image)
            state = processor.set_image(image)
            prompt_masks = []
            for prompt in prompts:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                prompt_mask = select_mask(
                    output,
                    image_array.shape[0] * image_array.shape[1],
                    args.min_area_ratio,
                    args.max_area_ratio,
                    args.selection,
                )
                if prompt_mask is None:
                    print(f"  prompt {prompt!r}: no retained candidates; skipping")
                else:
                    print(f"  prompt {prompt!r}: retained")
                    prompt_masks.append(prompt_mask)
            if not prompt_masks:
                raise RuntimeError(
                    "SAM3 produced no retained arm mask for any prompt. "
                    "Adjust --prompt, --confidence-threshold, or the "
                    "area-ratio limits."
                )
            foreground = np.logical_or.reduce(prompt_masks)
            binary_mask = foreground.astype(np.uint8) * 255
            save_png(binary_mask, mask_path, "L")
            if args.overlay_dir:
                overlay_path = args.overlay_dir.expanduser().resolve() / rgb_path.name
                save_overlay(image_array, binary_mask, overlay_path)

    missing_masks = [
        rgb_path.name.replace("rgb_", "arm_")
        for rgb_path in rgb_paths
        if not (sequence_dir / rgb_path.name.replace("rgb_", "arm_")).is_file()
    ]
    if missing_masks:
        print(
            f"Generated {len(pending_paths)} mask(s); {len(missing_masks)} remain. "
            "Run the same command again to resume."
        )
    else:
        if not args.no_update_metadata:
            set_arm_masks_available(metadata_path)
            print(f"Updated {metadata_path}: poor_arm_masks=false")
        print(f"All {len(rgb_paths)} arm masks are available in {sequence_dir}")


if __name__ == "__main__":
    main()
