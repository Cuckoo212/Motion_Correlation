# RealSense + SAM3 Cable 点云

采集对齐的 RGB-D，用 SAM3 的 `cable` 提示生成 Mask，再反投影为 Cable-only
点云，保存图像、元数据和逐帧 PLY。支持相机直连 GPU PC，或相机连接 Mini PC、
通过 ROS 2 将图像送至 GPU PC。

## 1. 代码结构

```text
/home/flexcycle/Motion_Correlation/cmcor/pointcloud/
├── __init__.py
├── realtime_cable_pc.py       # 相机直连：采集、SAM3、点云、保存和显示
├── ros2_capture_publisher.py  # Mini PC：采集、对齐、ROS 2 发布
├── ros2_frame_receiver.py     # GPU PC：ROS 2 接收、精确配对、缓存最新帧
├── remote_cable_pc.py         # GPU PC：获取接收帧、SAM3、点云、保存和显示
├── rgbd_transport.py          # 图像解码、帧配对、最新帧缓存和 Unix socket 协议
├── runtime.py                 # 共用录制、JSON 保存和 Matplotlib 显示
├── sam3_segmenter.py          # SAM3 分割封装
├── geometry.py               # Mask + Depth 反投影、体素下采样、PLY 保存
├── view_saved_cable_ply.py    # 查看已保存点云
├── requirements-realtime.txt # GPU 推理环境依赖
├── tests/test_rgbd_transport.py
└── sam3_project/              # 本地 SAM3 源码
    ├── sam3/
    ├── checkpoints/           # 默认权重：sam3.pt
    ├── pyproject.toml
    └── LICENSE
```

从仓库根目录使用 `python -m pointcloud.<模块>` 启动，无需 `colcon build`。
默认数据和权重路径随仓库位置确定；GPU PC 使用其实际仓库路径。

## 2. 环境准备

| 进程 | Python 环境 | 主要依赖 |
|---|---|---|
| Mini PC 采集发布 | ROS 2 系统 Python | rclpy、sensor_msgs、std_msgs、NumPy、OpenCV、pyrealsense2 |
| GPU PC 接收 | ROS 2 系统 Python | rclpy、sensor_msgs、std_msgs、NumPy |
| GPU PC 直连 / 远程推理 | 独立 SAM3 Conda 环境 | CUDA PyTorch、SAM3、OpenCV、NumPy、Pillow、Matplotlib；直连另需 pyrealsense2 |

当前 Mini PC 为 ROS 2 Humble，使用 `/usr/bin/python3`（Python 3.10），不要用
默认的 Python 3.13 启动 ROS 节点。GPU 上的 ROS 和 SAM3 分进程运行，避免混用环境。

GPU 原有已验证环境为 `realtime-cable-pc`（Python 3.12、PyTorch 2.7.0+cu126、
torchvision 0.22.0）。已有环境可直接使用；重建时先安装匹配 CUDA 的 PyTorch，
再在 `pointcloud/` 下安装 `requirements-realtime.txt`。权重放在
`pointcloud/sam3_project/checkpoints/sam3.pt`，或用 `--checkpoint` 指定。

下文 GPU 示例使用原主机路径 `/home/qihang/...`，请按实际路径替换。
启动采集前关闭 `realsense-viewer`、`realsense2_camera` 等占用相机的进程。

## 3. 模式 A：相机直连 GPU PC

```text
相机 USB → GPU PC：采集与对齐 → SAM3 → 点云 → 保存 / 显示
```

**执行顺序：连接相机 → 检查 GPU 环境与权重 → 启动原入口。**
所有命令在 GPU PC 执行，不需要 ROS 发布或接收节点。

```bash
cd /home/qihang/Motion_Correlation/cmcor
unset PYTHONPATH
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.realtime_cable_pc
```

默认显示 Matplotlib 点云窗口；纯 SSH 终端加 `--no-view`。
首次可加 `--max-frames 1 --no-view` 做单帧自检。
按 `Ctrl-C` 或关闭窗口停止，等待元数据保存完成。

## 4. 模式 B：相机连接 Mini PC，SSH 操作 GPU PC

```text
Mini PC：相机 USB → 采集、对齐 → ROS 2 DDS 网络
GPU PC：ROS 接收、配对 → 最新帧缓存 → 本机 Unix socket → SAM3、点云、保存
```

两台机器都需要最新代码，建议使用相同 ROS 2 发行版和有线千兆局域网。
**SSH 用于登录和启动命令，图像通过 ROS 2 DDS 传输。** SSH 能登录不代表
DDS 已连通；两台机器须允许 DDS 发现和数据通信，普通 SSH 隧道不能自动替代它。

### 步骤 1：Mini PC 终端 A，启动采集发布

```bash
cd /home/flexcycle/Motion_Correlation/cmcor
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
/usr/bin/python3 -m pointcloud.ros2_capture_publisher \
  --width 640 --height 480 --fps 30 --publish-fps 5
```

默认采集 30 Hz、最多发布 5 Hz，不在 Mini PC 保存文件。
可加 `--serial <相机序列号>`；需要本地原始 RGB-D 录制时，加
`--record-root /home/flexcycle/Motion_Correlation/cmcor/datasets/raw_rgbd`。
本地同步写 PNG，受磁盘与 CPU 影响，不保证完整保留 30 Hz 的每一帧。

### 步骤 2：终端 B，SSH 登录 GPU PC，启动接收

在 Mini PC 新终端执行：

```bash
ssh <GPU用户名>@<GPU地址>
```

登录后，以下命令在 **GPU PC** 执行：

```bash
cd /home/qihang/Motion_Correlation/cmcor
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
ros2 topic list
/usr/bin/python3 -m pointcloud.ros2_frame_receiver \
  --socket-path /tmp/cable_rgbd.sock
```

应看到 `/cable_camera/` 下的 `color/image_raw`、`aligned_depth/image_raw`、
`color/camera_info`、`metadata` 四个话题。接收日志每 5 秒显示配对计数，
确认持续增长后再启动推理。若未收到数据，检查域 ID、网卡、RMW 和防火墙。

### 步骤 3：终端 C，另开 SSH，启动 GPU 推理

在 Mini PC 另一个终端执行：

```bash
ssh <GPU用户名>@<GPU地址>
```

登录后，以下命令在 **GPU PC** 执行，不需要加载 ROS 环境：

```bash
cd /home/qihang/Motion_Correlation/cmcor
unset PYTHONPATH
/home/qihang/miniconda3/envs/realtime-cable-pc/bin/python \
  -m pointcloud.remote_cable_pc \
  --socket-path /tmp/cable_rgbd.sock --no-view
```

首次可加 `--max-frames 1 --sequence-name remote_smoke_test` 做单帧自检。
需要交互窗口时，在 GPU PC 本地桌面终端运行推理命令，去掉 `--no-view`。
接收与推理须使用相同 socket 路径，并以同一 GPU 用户运行。

### 步骤 4：停止

依次在 **GPU 推理终端 C → GPU 接收终端 B → Mini PC 采集终端 A** 按
`Ctrl-C`；先等待推理元数据保存完成。接收节点正常退出会清理 socket。
异常退出遗留 socket 时，确认旧进程已停止后再清理，或换一个 socket 路径。
断流超过默认 10 秒会结束推理并保存已完成帧，可用 `--receive-timeout` 调整；
恢复后使用新序列名重新启动推理。

## 5. 输出与常用参数

两种模式均在 GPU PC 仓库下保存：

```text
 datasets/realtime_cable_pc/<序列名>/
   rgb_00000000.png、depth_00000000.png、mask_00000000.png、…
   camera_info.json、frames.json、processing_stats.json
 output/realtime_cable_pc/<序列名>/
   cable_camera_00000000.ply、…
```

默认序列名为启动时间；`--sequence-name` 不可重复。同序号 RGB、Depth、Mask、
PLY 一一对应，Depth PNG 为 uint16 毫米，PLY 坐标单位为米；无有效点仍保存空 PLY。
相机信息记录内参和深度约定，帧索引与统计记录时间戳、推理耗时及点数。

| 参数 | 用途 |
|---|---|
| `--no-view` / `--view-rgb` | 关闭窗口 / 使用真实 RGB 点颜色 |
| `--max-frames 1` | 单帧自检 |
| `--checkpoint <路径>` | 指定 SAM3 权重 |
| `--selection union` / `best` | 合并候选 / 仅保留最高分实例 |
| `--min-depth-mm` / `--max-depth-mm` | 深度范围，默认 150–2000 mm |
| `--voxel-size-mm` | 下采样大小，默认 2 mm |
| `--dataset-root` / `--output-root` | 指定图像与点云保存根目录 |

各入口的全部参数使用 `--help` 查看。远程模式的分辨率、采集 / 发布频率在
Mini PC 发布端设置，远程推理不接受 `--reuse-last-mask`。

远程模式只处理最新完整帧对，跳过中间帧以避免积压；GPU 仅保存参与推理的帧。
若启用 Mini PC 录制，可通过 `timestamp_ns` 关联原始帧。640×480 原始 RGB-D
在 5 Hz 下约占 61 Mb/s（不含协议开销）。点云频率仍受 SAM3 速度限制：原
RTX 3060 Ti 在 640×480 下约 703 ms/帧、1.4 Hz，其他 GPU 需实测。
跨机输入 / 结果年龄统计需两台机器同步时钟，不能直接当作精确网络延迟。

## 6. 验证

无相机 / GPU 测试（在仓库根目录）：

```bash
/usr/bin/python3 -m unittest discover -s pointcloud/tests -v
```

已有验证：5 项传输与保存测试通过，Mini PC 真实 RealSense 的本机 ROS 2
采集、配对和取帧通过；跨机网络及目标 GPU 的实际 SAM3 推理仍需单帧自检。
