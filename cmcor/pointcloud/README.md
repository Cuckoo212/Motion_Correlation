# Realtime SAM3 cable-only point cloud

This first-stage pipeline captures aligned RealSense RGB-D, runs a SAM3
`cable` text prompt, projects the mask into a cable-only point cloud, and
optionally displays it with Open3D.

Outputs:

```text
datasets/realtime_cable_pc/<sequence>/
  rgb_DDDDDDDD.png
  depth_DDDDDDDD.png
  mask_DDDDDDDD.png
  camera_info.json
  frames.json

output/realtime_cable_pc/<sequence>/
  latest.ply
  latest_overlay.png
  processing_stats.json
  cable_DDDDDDDD.ply       # only with --save-cloud-every N
```

The package is intentionally located at the repository root and vendors the
SAM3 source below `pointcloud/sam3_project`. It does not import code from the
Franka workspace.

Run from the repository root in the SAM3 CUDA environment:

```bash
python -m pointcloud.realtime_cable_pc
```

Install Open3D for the interactive point-cloud window. Without it:

```bash
python -m pointcloud.realtime_cable_pc --no-view
```

The default checkpoint location is:

```text
pointcloud/sam3_project/checkpoints/sam3.pt
```

SAM3 checkpoints are access-controlled by their publisher and are not present
in the source tree copied from `cable_interact`. Put an authorized checkpoint
at that location or pass `--checkpoint /absolute/path/to/sam3.pt`.

Important: image-mode SAM3 is evaluated synchronously here, so the capture
rate equals the SAM inference rate. This is intentional for the first,
frame-exact implementation. A later recorder/perception split can capture
RGB-D at 30 Hz while SAM processes only the newest frame.
