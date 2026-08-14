# RealSense + SAM3 实时 Cable-only 点云

本模块使用 RealSense 采集对齐的 RGB-D 图像，通过 SAM3 的 `cable`
文本提示生成当前帧 Cable Mask，再将 Mask 内的深度像素反投影为
Cable-only 三维点云。系统可以在采集时使用 Matplotlib 三维散点窗口实时显示点云，
同时保存 RGB、Depth、Mask、相机内参和点云结果。

## 1. 已实现功能

- 采集 RealSense 彩色图像和深度图像。
- 将 Depth 对齐到 Color 坐标。
- 使用 SAM3 `cable` 文本提示对每个当前帧进行分割。
- 根据 Cable Mask 和对齐深度生成 Cable-only 点云。
- 对点云进行可选体素下采样。
- 使用与 `view_saved_cable_ply` 相同的 Matplotlib 三维交互窗口实时显示Cable点云。
- 支持鼠标旋转、滚轮缩放、XYZ等比例坐标轴、可调点大小和可选RGB颜色。
- 按帧保存 RGB、Depth 和 SAM3 Mask。
- 保存相机内参、畸变参数、深度比例和帧索引信息。
- 为每一帧保存一个与数据集序号严格对应的 ASCII PLY 点云。

## 2. Workspace 结构

`pointcloud` 是仓库顶层的独立 Python 包：

```text
/home/qihang/Motion_Correlation/cmcor/
├── cmcor/                         # Motion Correlation 算法
├── datasets/
├── output/
└── pointcloud/
    ├── __init__.py
    ├── geometry.py                # Mask + Depth 反投影和 PLY 保存
    ├── sam3_segmenter.py          # SAM3 Cable 分割封装
    ├── realtime_cable_pc.py       # 实时系统入口
    ├── requirements-realtime.txt
    └── sam3_project/              # 本地 SAM3 源码及依赖声明
        ├── sam3/
        ├── checkpoints/
        ├── pyproject.toml
        └── LICENSE
```

运行时不会从 `franka_ros2_ws` 导入 SAM3 代码。

## 3. 数据保存结构

每次启动默认创建一个以当前时间命名的序列目录：

```text
/home/qihang/Motion_Correlation/cmcor/
├── datasets/
│   └── realtime_cable_pc/
│       └── 2026-08-12-xxxxxx/
│           ├── rgb_00000000.png
│           ├── depth_00000000.png
│           ├── mask_00000000.png
│           ├── rgb_00000001.png
│           ├── depth_00000001.png
│           ├── mask_00000001.png
│           ├── camera_info.json
│           ├── frames.json
│           └── processing_stats.json
└── output/
    └── realtime_cable_pc/
        └── 2026-08-12-xxxxxx/
            ├── cable_camera_00000000.ply
            ├── cable_camera_00000001.ply
            └── cable_camera_DDDDDDDD.ply
```

`camera_info.json` 包含：

- 彩色相机内参矩阵 `camera_matrix`。
- 相机畸变系数及畸变模型。
- RGB-D 分辨率和帧率。
- RealSense 深度比例。
- 保存的 Depth PNG 以毫米为单位的说明。
- Depth 已对齐到 Color 坐标的标记。

`frames.json` 记录帧索引、时间戳、RGB/Depth 文件名、SAM3 推理时间、
Mask 文件名、SAM3 分数和点云点数。

每个 `cable_camera_DDDDDDDD.ply` 都与同序号的 RGB、Depth 和 Mask 一一对应。
例如 `cable_camera_00000025.ply` 来源于 `rgb_00000025.png`、
`depth_00000025.png` 和 `mask_00000025.png`。即使某帧没有有效Cable深度点，
程序也会保存合法的零顶点PLY，因此序号不会缺失。PLY使用米为坐标单位，格式为：

```text
ply
format ascii 1.0
element vertex <点数>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
```

## 4. SAM3 权重

默认权重位置为：

```text
/home/qihang/Motion_Correlation/cmcor/pointcloud/
  sam3_project/checkpoints/sam3.pt
```

SAM3 checkpoint 由上游提供方控制访问。获得授权后，请将权重放入上述默认
位置。也可以在运行时使用 `--checkpoint` 指定其他绝对路径。

## 5. 环境与依赖

实时点云程序已在下面的独立 Conda 环境中完成端到端验证：

```text
Miniconda： /home/qihang/miniconda3
环境名： realtime-cable-pc
Python：   3.12.13
解释器： /home/qihang/miniconda3/envs/realtime-cable-pc/bin/python
GPU：      NVIDIA GeForce RTX 3060 Ti（7.63 GiB可用显存）
NVIDIA驱动：580.173.02
CUDA runtime：12.6（由PyTorch wheel提供）
```

已验证的主要 Python 包版本：

| 依赖 | 版本 | 用途 |
|---|---:|---|
| `torch` | `2.7.0+cu126` | SAM3 CUDA 推理 |
| `torchvision` | `0.22.0+cu126` | SAM3 视觉操作 |
| `numpy` | `1.26.4` | 图像和点云数组 |
| `opencv-python` | `4.11.0.86` | RGB/Depth/Mask 读写 |
| `pyrealsense2` | `2.58.3.10794` | D456 采集和 RGB-D 对齐 |
| `matplotlib` | `3.11.1` | 实时和单帧三维交互点云窗口 |
| `Pillow` | `12.2.0` | SAM3 图像输入 |
| `timm` | `1.0.28` | SAM3 模型依赖 |
| `einops` | `0.8.2` | SAM3 Transformer 张量变换 |
| `decord` | `0.6.0` | SAM3 间接导入的视频依赖 |
| `pycocotools` | `2.0.11` | SAM3 间接导入的 Mask 工具 |
| `setuptools` | `80.10.2` | 提供 SAM3 仍使用的 `pkg_resources` |

### 5.1 使用现有环境

可以不激活 Conda，直接使用经过验证的绝对解释器路径：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python --version
```

也可以激活环境：

```bash
source /home/qihang/miniconda3/bin/activate realtime-cable-pc
```

### 5.2 从零重建环境

下面命令使用 `conda-forge`，不依赖 Anaconda 默认频道：

```bash
/home/qihang/miniconda3/bin/conda create -y \
  -n realtime-cable-pc \
  --override-channels \
  -c conda-forge \
  python=3.12 pip
```

安装PyTorch 2.7.0 CUDA 12.6版：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python -m pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

安装实时点云、SAM3、RealSense和Matplotlib依赖：

```bash
cd /home/qihang/Motion_Correlation/cmcor/pointcloud
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python -m pip install \
  -r requirements-realtime.txt
```

不要使用 `sudo pip`，也不要将依赖安装到 ROS 2 使用的系统 Python。

SAM3 上游源码的依赖声明不完整，本项目的
`requirements-realtime.txt` 已额外固定/补充：

- `setuptools<81`：SAM3 仍导入已废弃的 `pkg_resources`。
- `einops`、`decord`和`pycocotools`：SAM3模型构建链会间接导入。

### 5.3 启动前验证

```bash
unset PYTHONPATH
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python - <<'PY'
import torch
import pyrealsense2 as rs
import matplotlib

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("D456 devices:", len(rs.context().query_devices()))
print("Matplotlib:", matplotlib.__version__)
PY
```

正常情况应显示 `CUDA: True`、一块 NVIDIA GPU、至少一台 RealSense
设备和 Matplotlib `3.11.1`。

## 6. 最新运行命令

进入仓库根目录：

```bash
cd /home/qihang/Motion_Correlation/cmcor
```

运行前先关闭 `realsense-viewer`、ROS `realsense2_camera` 和其他占用D456的
进程。默认权重已位于 `pointcloud/sam3_project/checkpoints/sam3.pt`。

### 6.1 正式启动（Matplotlib 三维交互窗口）

```bash
cd /home/qihang/Motion_Correlation/cmcor
unset PYTHONPATH
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc
```

程序默认创建与 `pointcloud.view_saved_cable_ply` 相同的Matplotlib三维散点窗口。
按住鼠标左键拖动可旋转，滚轮可缩放。默认使用原工具的蓝色散点；增加
`--view-rgb` 可显示点云保存的RGB颜色。关闭窗口或在启动终端按 `Ctrl-C`
会结束采集，然后在对应datasets序列目录写出 `frames.json` 和
`processing_stats.json`。

默认窗口尺寸为 `1280 x 800`，点大小为 `5.0`。例如使用更大的窗口和更粗的点：

```bash
cd /home/qihang/Motion_Correlation/cmcor
unset PYTHONPATH
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --view-width 1600 \
  --view-height 1000 \
  --point-size 7
```

Cable 较细、点云比较稀疏时，建议将 `--point-size` 设置为 `6` 到 `12`；点数很多
时可以设置为 `2` 到 `4`。这些参数只改变显示，不影响 PLY 内容和现有保存方式。

### 6.2 先运行1帧无窗口自检

```bash
cd /home/qihang/Motion_Correlation/cmcor
unset PYTHONPATH
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --sequence-name smoke_test \
  --max-frames 1 \
  --no-view
```

每个 `--sequence-name` 只能使用一次；如果目录已存在，请更换新名称。

### 6.3 逐帧点云保存方式

正式启动命令默认会：

- 将每帧 RGB、Depth 和 Mask 保存到
  `datasets/realtime_cable_pc/<时间戳>/`。
- 将每帧对应的点云保存为
  `output/realtime_cable_pc/<时间戳>/cable_camera_DDDDDDDD.ply`。
- 输出目录只包含逐帧Cable点云，不再生成 `latest.ply`、
  `latest_overlay.png` 或 `processing_stats.json`。
- `camera_info.json`、`frames.json` 和 `processing_stats.json` 保存在对应的
  datasets序列目录中。

如果只采集和计算，不需要 Matplotlib 窗口：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc --no-view
```

指定其他权重路径：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --checkpoint /absolute/path/to/sam3.pt
```

只使用 SAM3 分数最高的 Cable 实例：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --selection best
```

合并 SAM3 保留的所有 Cable 候选：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --selection union
```

设置 Mask 候选的分数和面积范围：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --confidence-threshold 0.5 \
  --min-area-ratio 0.0001 \
  --max-area-ratio 0.5
```

设置深度范围和点云体素大小：

```bash
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc \
  --min-depth-mm 150 \
  --max-depth-mm 2000 \
  --voxel-size-mm 2.0
```

按 `Ctrl-C` 停止采集。如果启用Matplotlib窗口，关闭窗口也会结束采集。

### 6.4 现代窗口参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--view-width` | `1280` | 点云窗口宽度 |
| `--view-height` | `800` | 点云窗口高度 |
| `--point-size` | `8.0` | 实时显示的点大小 |
| `--view-rgb` | 关闭 | 使用真实RGB颜色；默认显示蓝色散点 |
| `--no-view` | 关闭 | 完全禁用 GUI，计算和保存继续运行 |

窗口显示Camera frame下的等比例XYZ坐标轴和当前点数。该变化只替换显示层，
不改变RealSense采集、SAM3分割、点云反投影或逐帧PLY保存流程。

Matplotlib适合当前低频Cable-only稀疏点云，但不适合数十万点或10–30 Hz更新。
如果以后显示完整场景点云或显著提高推理频率，应改用Open3D、RViz2或Foxglove。

## 7. 当前处理时序和频率

当前第一版为同步、逐帧处理：

```text
采集一对对齐 RGB-D
  → SAM3 生成当前帧 Mask
  → 保存 RGB、Depth 和 Mask
  → Mask + Depth 反投影为 Cable-only 点云
  → 保存并显示点云
  → 采集下一帧
```

因此同一索引的三个文件严格对应：

```text
rgb_00000025.png
depth_00000025.png
mask_00000025.png
```

当前实际采集频率由 SAM3 单帧推理速度决定，不保证达到 RealSense 的30 Hz。
2026-08-13在这台RTX 3060 Ti电脑上的实测结果为：

```text
分辨率：640 x 480
SAM3平均单帧耗时：约703 ms
Cable Mask/点云更新频率：约1.4 Hz
```

RTX 3090上的频率需要在对应主机上另行实测，不能直接使用上述3060 Ti数据。

这种实现首先保证 RGB、Depth 和 Mask 不会错配。如果后续需要完整保留30 Hz
RGB-D，应将处理拆成异步流水线：

```text
RealSense 采集线程：30 Hz 持续录制 RGB-D
SAM3 推理线程：只处理最新的待分割帧
点云线程：对应 SAM3 结果更新 Cable-only 点云
```

该异步方案中，原始 RGB-D 可以保持30 Hz录制，Cable-only 点云则以 SAM3
实际推理频率更新。

## 8. 参数查看

查看全部参数：

```bash
cd /home/qihang/Motion_Correlation/cmcor
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc --help
```

## 9. 注意事项

- RealSense Depth 必须对齐到 Color，否则 Mask 像素不能直接对应深度像素。
- `union` 适合获取画面中所有Cable；`best` 只保留SAM3分数最高的实例。
- 当多根 Cable 互相重叠时，SAM3 的 `cable` 文本提示不一定能判断机器人实际
  抓住的是哪一根 Cable；这需要后续 Motion Correlation 结果进一步筛选。
- 每帧都会写出ASCII PLY；长时间运行时，输出体积会明显大于原来的二进制
  `latest.ply`，运行前应检查剩余磁盘空间。
- 长时间保存每帧 PNG 会产生较大磁盘占用，运行前应检查剩余空间。
