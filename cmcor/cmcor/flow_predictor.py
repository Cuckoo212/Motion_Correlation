"""
Search for and import an OnlineFlow optical flow estimator class.
"""

import os
import sys


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
folder_list = [
    os.path.join(repo_root, "movingcables", "flow_predictors"),
    ]

for folder in folder_list:
    if os.path.isdir(folder):
        sys.path.append(folder)

try:
    from online_flow import OnlineFlow as MaskFlowOnlineFlow
    MASK_FLOW_IMPORT_ERROR = None
except ImportError as e:
    MaskFlowOnlineFlow = None
    MASK_FLOW_IMPORT_ERROR = e
    print("WARNING: Cannot import OnlineFlow flow predictor!", e)

try:
    from online_flow_farneback import OnlineFlow as FarnebackOnlineFlow
    FARNEBACK_IMPORT_ERROR = None
except ImportError as e:
    FarnebackOnlineFlow = None
    FARNEBACK_IMPORT_ERROR = e
    print("WARNING: Cannot import Farneback flow predictor!", e)


class FlowPredictor():
    def __init__(self, gpu):
        if MaskFlowOnlineFlow is not None:
            try:
                self.mfnprob = MaskFlowOnlineFlow(
                    gpu=gpu, probabilistic=True, finetuned=True)
                self.backend = "maskflownet"
                return
            except Exception as e:
                print(
                    "WARNING: MaskFlownet OnlineFlow failed to initialize; "
                    f"falling back to Farneback flow. Error: {e}")

        if FarnebackOnlineFlow is not None:
            self.mfnprob = FarnebackOnlineFlow(gpu=gpu)
            self.backend = "farneback"
            return

        raise RuntimeError(
            "No optical flow predictor is available. "
            f"MaskFlownet import error: {MASK_FLOW_IMPORT_ERROR}; "
            f"Farneback import error: {FARNEBACK_IMPORT_ERROR}")

    def flow(self, img_target, img_reference):
        flow_result = self.mfnprob.flow(img_target, img_reference)
        return flow_result[0]
