# Motion_Correlaion

Minimal workspace for continuing CMCor-based motion correlation segmentation with a custom depth-camera setup.

This repository keeps the code needed to run CMCor with the MovingCables/MaskFlownet optical-flow backend, while excluding local datasets, generated outputs, virtual environments, ROS build products, and large pretrained weights.

## What is included

- `cmcor/`: CMCor source code, project README, requirements, and utilities.
- `movingcables/flow_predictors/`: wrapper used by CMCor to import `OnlineFlow`.
- `movingcables/MaskFlownet/`: minimal MaskFlownet runtime code and checkpoint metadata.
- `docs/dataset_format.md`: notes for preparing your own motion-correlation sequences.

## What is intentionally excluded

- CMCor datasets and annotations.
- Your own captured RGB/depth datasets.
- Runtime outputs, masks, plots, videos, point clouds, and temporary captures.
- Python/conda/ROS build environments.
- MaskFlownet `.params` model weights.

Place the MaskFlownet checkpoint locally at:

```text
movingcables/MaskFlownet/weights/b1aApr25-1426_320000.params
```

The weight file is required to run optical flow, but it should not be committed to git.

## Custom dataset sketch

For a new cable segmentation scene, create one folder per interaction sequence. A sequence should contain ordered RGB frames and gripper/action metadata. Depth frames are useful for your own pipeline but CMCor's motion correlation path mainly consumes RGB frames and gripper motion.

See `docs/dataset_format.md` for the expected structure.
