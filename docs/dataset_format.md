# Dataset Format Notes

CMCor's motion-correlation evaluation reads a list of sequence names, then loads each sequence folder from `motion_correlation_buffers`.

A practical custom layout is:

```text
datasets/MyWorkspace/
  my_sequences.json
  motion_correlation_buffers/
    2026-xx-xx-hhmmss/
      rgb_00000000.png
      rgb_00000001.png
      ...
      actions_gripper.json
      depth_00000000.png        # optional for your own downstream work
      arm_00000000.png          # optional; not needed when poor_arm_masks is true
```

## Frames

Each numbered RGB image is one time step in the interaction. The last RGB frame is used as the target image sample, and previous frames are compared to it through optical flow.

## Gripper/action metadata

`actions_gripper.json` provides the end-effector/gripper motion history. When you do not have robot arm segmentation masks, configure the sequence with `poor_arm_masks: true` so the method estimates robot/gripper motion from the gripper points instead of requiring `arm_*.png` masks.

## Masks

Ground-truth cable masks such as `cable_mask_*.png` are used for evaluation and parameter optimization. They are not required to produce a segmentation mask on new data.

## Recommended capture procedure

1. Keep the depth camera fixed relative to the workspace.
2. Record RGB frames during one grasp/pull/manipulation action.
3. Save frames with stable zero-padded indices.
4. Save gripper or end-effector positions for the same time steps when possible.
5. Use `poor_arm_masks: true` unless you have reliable robot arm masks.
