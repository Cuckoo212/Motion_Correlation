"""
The main configuration file of the cmcor code package.
"""

import os
import json



class MotionPerceptionSettings():
    gpu = False
    
    motion_correlation_min_a = 0.18   # Interactivity threshold.
    motion_correlation_min_cod = 0.86 # Coefficient of determination threshold.
    motion_segmentation_vote_thr = 47 # Relative. Between 0 and 100.
    dataset_root = "/home/flexcycle/cv_models/cmcor/datasets/CMCor"
    output_root = "/home/flexcycle/cv_models/cmcor/output"
    double_grasp_sequences = [
        # 2024 double grasp sets
        ["2024-10-30-161137", "2024-10-30-161441"],
        ["2024-10-30-162624", "2024-10-30-163157"],
        ["2024-11-07-143845", "2024-11-07-144321"],
        ["2024-11-07-145204", "2024-11-07-145507"],
        ["2024-11-07-151007", "2024-11-07-151317"],
        ["2024-11-07-161403", "2024-11-07-161718"],
        ["2024-11-07-162911", "2024-11-07-163128"],
        ["2024-11-07-170848", "2024-11-07-171207"],
        ["2024-11-07-171626", "2024-11-07-171904"],
        ["2024-11-07-172233", "2024-11-07-172445"],
        # 2025-03-20 multi grasp sets
        [
            "2025-03-20-135150", "2025-03-20-135445",
            "2025-03-20-135709", "2025-03-20-135946",
            ],
        [
            "2025-03-20-140239", "2025-03-20-140455",
            "2025-03-20-140909", "2025-03-20-141410", "2025-03-20-141725",
            ],
        ["2025-03-20-142304", "2025-03-20-142525"],
        ["2025-03-20-145552", "2025-03-20-145824", "2025-03-20-150125"],
        [
            "2025-03-20-150512", "2025-03-20-151300",
            "2025-03-20-151659", "2025-03-20-152036",
            ],
        [
            "2025-03-20-163147", "2025-03-20-163423", "2025-03-20-163719",
            "2025-03-20-164015", "2025-03-20-164341", "2025-03-20-164616",
            ],
        # 2025-04-09 multi grasp sets
        [
            "2025-04-09-134352", "2025-04-09-134811", "2025-04-09-135123",
            "2025-04-09-135613", "2025-04-09-135830", "2025-04-09-140332",
            ],
        [
            "2025-04-09-142225", "2025-04-09-142728", "2025-04-09-143100",
            "2025-04-09-143328", "2025-04-09-143549",
            ],
        [
            "2025-04-09-145759", "2025-04-09-150024", "2025-04-09-150247",
            "2025-04-09-150606", "2025-04-09-151012",
            ],
        [
            "2025-04-09-151759", "2025-04-09-152033", "2025-04-09-152315",
            "2025-04-09-152935", "2025-04-09-153444", "2025-04-09-153659",
            ],
        ]
    double_grasp_params = [
        [90, 7],
        [90, 7],
        [87, 19],
        [87, 19],
        [87, 19],
        [87, 19],
        [87, 19],
        [87, 19],
        [87, 19],
        [87, 19],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        [86, 18],
        ]
    beta_to_plot = 0.4
    eval_all_betas = [1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    eval_params_segmentation = [31, 47, 47, 47, 50, 52]
    eval_params_correlation = [
        [66, 17], [77, 20], [86, 18], [92, 36], [92, 48], [94, 50]]
    eval_params_flow_std = [26, 32, 34, 35, 38, 47]

    def __init__(self):
        self.data_root = os.path.join(
            self.dataset_root, "motion_correlation_buffers")
        self.gt_root = os.path.join(
            self.dataset_root, "motion_correlation_annotations")
        self.multigrasp_json_path = os.path.join(
            self.dataset_root, "multigrasp_sequences.json")
        if os.path.isfile(self.multigrasp_json_path):
            with open(self.multigrasp_json_path, "r") as fp:
                print(f"Loading {self.multigrasp_json_path}...")
                self.double_grasp_sequences = json.load(fp)
                assert(
                    len(self.double_grasp_sequences) <=
                    len(self.double_grasp_params))
                self.double_grasp_params = self.double_grasp_params[
                    0:len(self.double_grasp_sequences)]
