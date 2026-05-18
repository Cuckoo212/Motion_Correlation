"""
Run double-grasp motion correlation and segmentation on the CMCor dataset.
Save the results to disk.
"""

import os
import pathlib
import argparse

import tqdm

from cmcor import motion_perception
from cmcor import motion_perception_cli
from cmcor import motion_perception_settings



def get_multi_name(sequence_names):
    prefix = os.path.commonprefix(sequence_names)
    multi_name_list = [prefix]
    for name in sequence_names:
        multi_name_list.append(name[len(prefix):])
    multi_name = '-'.join(multi_name_list)
    return multi_name


def main(folder_in, sequences, folder_out, folder_plots, algorithm, gpu=True):
    mp = motion_perception.MotionPerception(gpu)
    for seq in tqdm.tqdm(sequences):
        out_name = get_multi_name(seq)
        seq_out = os.path.join(folder_out, out_name)
        pathlib.Path(seq_out).mkdir(parents=True, exist_ok=True)
        sequence_folders = [os.path.join(folder_in, name) for name in seq]
        mpm = motion_perception_cli.MotionPerceptionMulti(
            seq_out, sequence_folders, gpu, algorithm, mp=mp)
        mpm.main()
        img_out = os.path.join(folder_plots, f"{out_name}.png")
        #motion_perception_cli.view_segmentation(
        #    seq_out, algorithm, img_out)


if __name__ == "__main__":
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    out_root = mp_settings.output_root
    folder_in = mp_settings.data_root
    gpu = mp_settings.gpu
    folder_out_correlation = os.path.join(out_root, "correlation_masks_multi")
    folder_plots_correlation = os.path.join(out_root, "correlation_plots_multi")
    sequences = mp_settings.double_grasp_sequences
    main(
        folder_in, sequences, folder_out_correlation,
        folder_plots_correlation, "correlation", gpu)

