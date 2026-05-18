"""
Run single-grasp motion correlation and segmentation on the CMCor dataset.
Save the results to disk.
"""

import os
import pathlib
import argparse

import tqdm

from cmcor import motion_perception
from cmcor import motion_perception_cli
from cmcor import motion_perception_settings



def get_clip_names(folder_in):
    clip_names = []
    for name in sorted(os.listdir(folder_in)):
        if not name.startswith("20"):
            continue
        folder_seq = os.path.join(folder_in, name)
        if not os.path.isdir(folder_seq):
            continue
        clip_names.append(name)
    return clip_names


def main(
        folder_in, clip_names, folder_out, folder_plots, algorithm, gpu=True,
        progress_bar=None):
    mp = motion_perception.MotionPerception(gpu)
    for name in clip_names:
        folder_seq = os.path.join(folder_in, name)
        if not os.path.isdir(folder_seq):
            continue
        seq_out = os.path.join(folder_out, name)
        pathlib.Path(seq_out).mkdir(parents=True, exist_ok=True)
        sequence_folders = [folder_seq]
        mpm = motion_perception_cli.MotionPerceptionMulti(
            seq_out, sequence_folders, gpu, algorithm, mp=mp)
        mpm.main()
        img_out = os.path.join(folder_plots, f"{name}.png")
        #motion_perception_cli.view_segmentation(
        #    seq_out, algorithm, img_out)
        if progress_bar is not None:
            progress_bar.update(1)


if __name__ == "__main__":
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    out_root = mp_settings.output_root
    folder_in = mp_settings.data_root
    gpu = mp_settings.gpu
    folder_out_correlation = os.path.join(out_root, "correlation_masks")
    folder_out_segmentation = os.path.join(out_root, "segmentation_masks")
    folder_plots_correlation = os.path.join(out_root, "correlation_plots")
    folder_plots_segmentation = os.path.join(out_root, "segmentation_plots")
    clip_names = get_clip_names(folder_in)
    tqdm_total = len(clip_names)*2
    progress_bar = tqdm.tqdm(total=tqdm_total, position=0, leave=False)
    main(
        folder_in, clip_names, folder_out_correlation,
        folder_plots_correlation, "correlation", gpu, progress_bar)
    main(
        folder_in, clip_names, folder_out_segmentation,
        folder_plots_segmentation, "segmentation", gpu, progress_bar)

