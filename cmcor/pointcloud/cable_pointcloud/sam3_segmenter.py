"""Reusable SAM3 text-prompt cable segmenter."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class Sam3CableSegmenter:
    def __init__(
        self,
        sam3_project: Path,
        checkpoint: Path,
        *,
        prompt: str = "cable",
        confidence_threshold: float = 0.5,
        min_area_ratio: float = 0.0001,
        max_area_ratio: float = 0.5,
        selection: str = "union",
        morphology_kernel: int = 3,
    ) -> None:
        sam3_project = sam3_project.expanduser().resolve()
        checkpoint = checkpoint.expanduser().resolve()
        if not (sam3_project / "sam3").is_dir():
            raise FileNotFoundError(f"SAM3 source not found: {sam3_project}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
        if selection not in {"union", "best"}:
            raise ValueError("selection must be 'union' or 'best'")
        if not 0 <= min_area_ratio <= max_area_ratio <= 1:
            raise ValueError("area ratios must satisfy 0 <= min <= max <= 1")

        sys.path.insert(0, str(sam3_project))
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 realtime segmentation requires a CUDA GPU")
        self.torch = torch
        self.prompt = prompt
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.selection = selection
        self.morphology_kernel = morphology_kernel
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device="cuda",
        )
        self.processor = Sam3Processor(
            model,
            device="cuda",
            confidence_threshold=confidence_threshold,
        )

    def segment(self, rgb_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Return a uint8 mask and the best retained candidate score."""
        image = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        with self.torch.inference_mode():
            state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(
                state=state,
                prompt=self.prompt,
            )

        area = rgb_bgr.shape[0] * rgb_bgr.shape[1]
        candidates: list[tuple[float, np.ndarray]] = []
        for mask, score in zip(output["masks"], output["scores"]):
            candidate = np.squeeze(mask.detach().cpu().numpy()).astype(bool)
            ratio = float(candidate.sum()) / float(area)
            if self.min_area_ratio <= ratio <= self.max_area_ratio:
                candidates.append((float(score.detach().cpu().item()), candidate))
        if not candidates:
            return np.zeros(rgb_bgr.shape[:2], np.uint8), 0.0

        candidates.sort(key=lambda item: item[0], reverse=True)
        foreground = (
            candidates[0][1]
            if self.selection == "best"
            else np.logical_or.reduce([candidate for _, candidate in candidates])
        )
        mask = foreground.astype(np.uint8) * 255
        if self.morphology_kernel > 1:
            kernel = np.ones(
                (self.morphology_kernel, self.morphology_kernel),
                np.uint8,
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, candidates[0][0]
