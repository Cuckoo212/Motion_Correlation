# cable_pointcloud：ROS 2 Humble 实时 Cable 点云

ROS 包和 Python 模块统一命名为 `cable_pointcloud`，旧 `pointcloud` Python 模块、
独立直连入口及 TCP 点云回传代码已删除。外层 `pointcloud/` 仅作为包源码目录保留，
以保持权重和挂载路径稳定。GPU 节点全部在一个 Humble + SAM3 Docker 容器内运行。

## 1. 四节点流程

```text
Mini PC                             GPU PC：Humble + SAM3 Docker
① rgbd_capture_publisher ── RGB-D ─→ ② sam3_pointcloud
                                     ├─ 接收线程：订阅、时间戳配对、最新帧缓存
                                     └─ 推理线程：SAM3、点云计算和保存
                                             ↓ PointCloud2
④ pointcloud_viewer ←───────────── ③ pointcloud_publisher
```

| 节点 | 运行位置 | 职责 |
| --- | --- | --- |
| `rgbd_capture_publisher` | Mini PC | 采集、对齐 RGB-D，发布图像、相机信息和实测吞吐统计 |
| `sam3_pointcloud` | GPU Docker | 直接订阅图像，独立接收线程配对并缓存最新帧；推理线程分割、生成和保存点云 |
| `pointcloud_publisher` | GPU Docker | 将内部 PointCloud2 话题转发回 Mini PC |
| `pointcloud_viewer` | Mini PC | 订阅点云并显示 Matplotlib 窗口 |

GPU 使用同一个 Humble + SAM3 容器，已移除独立 receiver 节点和 Unix socket。
SAM3 直接从进程内最新帧缓存读取数组，不通过磁盘或 socket。
Mini PC 仍将 RGB 和 Depth 分成两个话题发送，因此按相同时间戳配对以避免跨帧混用。
接收线程持续更新单帧缓存，推理较慢时主动跳过旧帧，不阻塞订阅回调。
接收、配对和两端吞吐统计都由 sam3_pointcloud 节点打印，原有频率日志继续保留。
默认只计算与发布，不保存录制数据；显式启用 record 后同步保存图像、mask、PLY。
点云回传采用标准 ROS 2 DDS 话题。

## 2. 代码结构

```text
pointcloud/                         # ROS 包源码根目录
├── package.xml                     # ament_python 包声明
├── setup.py / setup.cfg             # 安装和 ros2 run 入口
├── resource/cable_pointcloud        # ament 包索引
├── cable_pointcloud/                # 唯一 Python 模块
│   ├── __init__.py
│   ├── rgbd_capture_publisher.py     # ① 相机采集发布
│   ├── ros_rgbd_input.py             # 推理节点内的图像订阅、配对和吞吐统计
│   ├── ros_nodes.py                  # ②③ ROS 入口和 PointCloud2 编解码
│   ├── inference.py                  # SAM3、点云保存和频率统计
│   ├── inference_args.py             # 推理参数
│   ├── sam3_segmenter.py             # SAM3 封装
│   ├── rgbd_transport.py             # 最新帧缓存和本地协议
│   ├── geometry.py                   # 反投影、下采样、PLY
│   └── runtime.py                    # JSON 和 Matplotlib 显示
├── launch/
│   ├── mini_pc.launch.py             # ①④
│   └── gpu_pc.launch.py              # ②③
├── docker/
│   ├── Dockerfile.sam3               # Humble + SAM3 单镜像
│   ├── compose.yaml                  # 唯一 gpu 服务
│   ├── ros-entrypoint.sh             # 自动加载 ROS 环境
│   └── build-sam3.sh                 # 安装 GPU 容器运行时、构建、自检
├── requirements-realtime.txt
├── tests/
└── sam3_project/                     # 独立 SAM3 源码和权重
    ├── COLCON_IGNORE                # 不作为 ROS 包构建
    ├── sam3/
    └── checkpoints/sam3.pt           # 权重不提交 Git
```

## 3. GPU PC 构建：在宿主机执行

先停止旧推理和采集。旧容器若还在运行，用 `sudo docker ps`
找到容器名，再用 `sudo docker stop <容器名>` 停止，避免重复计算和发布。

```bash
cd /home/flexcycle/qihang/Motion_Correlation/cmcor/pointcloud/docker
sudo docker compose build gpu
```

镜像使用 Ubuntu 22.04、ROS 2 Humble、Python 3.10 SAM3 venv、PyTorch 2.7.0+cu128。
构建时安装 requirements-realtime.txt，并执行 colcon build。权重从宿主机挂载，
不打包到镜像中。修改代码后需重建镜像。

宿主机须已配置 NVIDIA Container Toolkit；首次安装可执行以下脚本，注意它会重启 Docker：

```bash
bash build-sam3.sh
```

## 4. Mini PC 构建：在 Mini PC 执行

同步新包源码，无需复制 GPU 权重。以下仓库路径按实际位置替换。
使用 Humble 系统 Python，保留已有 pyrealsense2。

```bash
sudo apt install python3-colcon-common-extensions python3-opencv python3-matplotlib
source /opt/ros/humble/setup.bash
cd /home/flexcycle/Motion_Correlation/cmcor
colcon build --base-paths pointcloud --packages-select cable_pointcloud
source install/setup.bash
/usr/bin/python3 -c 'import pyrealsense2'
```

直接使用仓库内的 `cmcor` 作为工作空间，无需 `cable_ws` 或源码软链接。
构建生成 `cmcor/build/`、`cmcor/install/`、`cmcor/log/`，已有 Git 忽略规则覆盖这些目录。
不要加载旧工作空间的安装环境；如 shell 启动文件中曾添加旧 setup.bash，请删除对应行并新开终端。

GPU 镜像也使用 `cmcor` 工作空间，位于 `/opt/cmcor`，入口加载
`/opt/cmcor/install/setup.bash`。运行时 `/workspace/cmcor` 挂载宿主机项目及数据；
安装结果单独放在 `/opt/cmcor`，避免宿主机挂载覆盖镜像内已构建的 ROS 包。

## 5. 启动

### Mini PC：本地桌面终端

关闭占用相机的程序。一次启动采集发布和点云显示：

```bash
source /opt/ros/humble/setup.bash
source /home/flexcycle/Motion_Correlation/cmcor/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch cable_pointcloud mini_pc.launch.py \
  publish_fps:=30 \
  crop_left:=100 crop_top:=140 \
  crop_width:=400 crop_height:=300
```

相机采集 30 Hz；publish_fps 可设为 5、10 或 30 Hz。配置值不保证实际频率。

### GPU PC：先进入 Docker，再在容器内启动

在 GPU 宿主机终端或 SSH 终端执行：

```bash
cd /home/flexcycle/qihang/Motion_Correlation/cmcor/pointcloud/docker
# 构建镜像
sudo docker compose build gpu

# 使用镜像创建并进入容器
sudo docker compose run --rm --no-deps gpu bash

为了以后方便，可以在第一个终端指定固定名称：
sudo docker compose run --rm --no-deps --name cable-gpu gpu bash
第二个终端直接进入：
sudo docker exec -it cable-gpu bash
```
创建容器时，入口脚本 `ros-entrypoint.sh` 自动加载 Humble 和
`/opt/cmcor/install/setup.bash`，无需手动 source。以下命令在容器内执行：

```bash
ros2 pkg executables cable_pointcloud
ros2 launch cable_pointcloud gpu_pc.launch.py
```

# 连续运行指令
```bash
bash -c 'trap "exit 0" INT TERM; while true; do ros2 launch cable_pointcloud gpu_pc.launch.py; echo "进程已退出，立即重启…"; done'
```


一次启动 SAM3 直接订阅计算和点云发布；GPU 无需桌面窗口，在 Mini PC 查看点云。
输入等待及断流超时默认 60 秒，建议先启动 Mini PC。

也可以在宿主机直接启动容器中的 launch，无需先进入 bash：

```bash
sudo docker compose run --rm --no-deps gpu
```

两种方式二选一，不要同时重复启动。

## 6. 检查、输出和停止

在 Mini PC 已加载 ROS 包的终端或 GPU 容器内执行：

```bash
ros2 node list
ros2 topic list
ros2 topic hz /cable_camera/cable_points
```

RGB-D 输入话题为 `/cable_camera/` 下的 color/image_raw、aligned_depth/image_raw、
color/camera_info、metadata。内部点云为 `/cable_pointcloud/internal`，最终回传点云为
`/cable_camera/cable_points`，坐标单位为米，保留相机 frame_id 和采集时间戳。
点云接收使用 BEST_EFFORT QoS，显示最新结果。

GPU 终端每约 5 秒打印配置采集/发布频率、实际接收/点云频率、SAM3 平均耗时及跳过帧数。
默认运行不会创建录制目录或写入图像、mask、PLY、JSON；终端日志和点云发布不受影响。
默认模式使用常量大小的累计统计，不保留随帧数增长的录制记录列表。

需要录制时，在 GPU 容器内显式执行：

```bash
ros2 launch cable_pointcloud gpu_pc.launch.py record:=true
```

默认不保存的命令仍为：

```bash
ros2 launch cable_pointcloud gpu_pc.launch.py
```

也可在宿主机通过 Docker 直接启动录制：

```bash
sudo docker compose run --rm --no-deps gpu ros2 launch cable_pointcloud gpu_pc.launch.py record:=true
```

单独使用 ros2 run 启动计算节点时，加入 `--record` 开启保存（仍需独立点云发布节点）。
仅指定 dataset-root、output-root 或 sequence-name 不会开启录制。
开启录制后，图像、mask、元数据和 PLY 保存在 GPU 宿主机：

```text
cmcor/datasets/realtime_cable_pc/<序列名>/
cmcor/output/realtime_cable_pc/<序列名>/
```

先在 GPU launch 终端按 Ctrl-C；开启录制时等待元数据保存完成，再停止 Mini PC launch。
若进入 bash 启动，最后输入 exit 退出容器。配对计数不增长时检查发布节点、域 ID、
RMW、网络及防火墙。

## 7. 测试

### 原生崩溃诊断

Compose 已设置 `PYTHONFAULTHANDLER=1`。新建容器后，SIGSEGV 等原生崩溃会尽可能
在终端输出各 Python 线程的调用栈。它是诊断功能，不是修复；通常无需重建镜像。
若正在使用旧容器，请退出后重新创建。

怀疑 CUDA 异步错误时，在宿主机临时执行：

```bash
sudo docker compose run --rm --no-deps -e CUDA_LAUNCH_BLOCKING=1 gpu
```

这会改变 CUDA 执行时序并降低速度，仅用于诊断，不用于频率对比。
同时在 GPU 宿主机检查 `journalctl -k` 中的 segfault、NVRM/Xid 或 OOM 记录。
libcuda 中的段错误需要结合崩溃栈定位，不能仅凭 exit code -11 判定显存不足或网络断流。

### 传输诊断日志

运行 GPU launch 时，SAM3 节点内的接收线程每约 5 秒在同一终端打印：

```text
[transport MiniPC] capture=29.90Hz publish=12.90Hz captured_total=... published_total=... stats_age=... session=...
[transport GPU] rgb=12.80Hz depth=12.80Hz paired=12.78Hz rgb_total=... depth_total=... paired_total=... metadata_wait_total=... invalid_total=... pending=... window=... last_pair_age=...
```

频率为各端最近统计窗口的实际计数/单调时钟时长；total 为各自进程启动后的累计值。
Mini PC 的 published 表示一对 RGB-D 的两个 publish 调用均已返回，不代表网络确认。
`/cable_camera/transport_stats` 将 Mini PC 实测计数发送给 GPU，无需跨机时钟同步。
两端窗口和启动时间不完全一致，不应直接相减累计值推算丢包；重启发布端会更换 session。
stats_age 是 GPU 收到最新统计后的本地时间，超过 12 秒标记 STALE。
Mini PC 未更新时显示 N/A，不影响原图像接收。

publish 明显低于配置值时检查采集发布；publish 高但 GPU rgb/depth 低时检查 DDS/网络；
rgb/depth 高而 paired 低时检查配对、metadata_wait、invalid 和 pending。
metadata_wait/invalid 统计消息数量，不等于丢失帧对数量。
与推理的 `[rates]` 累计平均日志一起使用，可区分输入吞吐与 SAM3 计算瓶颈。
更新后 GPU 需重建镜像，Mini PC 需同步源码并在 cmcor 重新 colcon build。

### 自动测试

在包源码根目录加载 ROS 环境后执行：

```bash
python3 -m unittest discover -s tests -v
```

完整测试依赖 SAM3 运行依赖及 rclpy；真实相机、Humble 镜像和跨机通信需在目标环境验证。


### 段错误诊断（2026-10-23）

逐帧日志现在强制刷新，并记录 `wait`（等待新 RGB-D）、`geometry`、
`publish`（消息转换及发布调用）、`publish_gap`（两次发布调用完成的间隔）、
`done_monotonic`。这些指标不表示 Mini PC 已收到消息。
首帧 `publish_gap=N/A`。录制开启时，保存文件的耗时也会计入发布间隔。

在已安装 GDB 的 GPU 容器中运行工作区源码（不需要先重建镜像）：

```bash
bash /workspace/cmcor/pointcloud/tools/debug_gpu_pc.sh --max-frames 5000
```

该脚本只运行推理节点，发布内部话题，不启动点云转发节点。
日志写入宿主机挂载的 `pointcloud/diagnostics/`。
同步定位可在脚本前设置 `CUDA_LAUNCH_BLOCKING=1`；这会影响性能，仅用于诊断。
GDB 的批处理退出状态不能代替检查日志中的 SIGSEGV/正常退出。

不启动 ROS，使用历史 RGB 图片隔离 SAM3：

```bash
export PYTHONPATH=/workspace/cmcor/pointcloud:${PYTHONPATH:-}
/opt/sam3-env/bin/python /workspace/cmcor/pointcloud/tools/stress_sam3.py \
  /workspace/cmcor/datasets/realtime_cable_pc/2026-10-09-162716 --frames 5000
```

已在真实 RGB-D 推理和单张历史图片循环中捕获到相同的
`libcuda.so.1 -> cuLaunchKernel -> torch.cat` 段错误。
因此 ROS/网络不是复现此问题的必要条件；具体底层根因仍需对照确认。
诊断日志保存在工作区顶层 `diagnostics/gpu_pc_2026-10-23/`。
不要把自动重启或同步模式下短时不崩溃当成根因已修复。


### 有线点云接收修复（2026-10-26）

GPU Compose 默认加载 `config/fastdds/gpu_pc_mtu1400.xml`，限制 UDP 报文大小并显式
设置 4 MiB 接收缓冲区。Mini PC 需要采用对应的有线 MTU1400 配置。
正式 `/cable_camera/cable_points` 输出默认使用 RELIABLE、KEEP_LAST depth=5。
内部 RGB-D 和内部点云的 QoS 保持原设置。发布可靠数据不等于保证实时频率；
网络拥塞或重传可能增加延迟，仍需持续测量。

```bash
ros2 launch cable_pointcloud gpu_pc.launch.py
# 回退正式输出为 BEST_EFFORT：
ros2 launch cable_pointcloud gpu_pc.launch.py cloud_reliability:=best_effort
```

Mini PC 在加载其 XML 的终端中验证正式话题：

```bash
ros2 topic echo /cable_camera/cable_points --qos-reliability reliable --once --field width
```

RViz 的对应 PointCloud2 显示项也应选择 Reliability=Reliable。
自定义订阅程序必须设置 RELIABLE 才能使用可靠重传；BEST_EFFORT 订阅者虽能匹配
RELIABLE 发布者，但不会因此自动得到可靠传输保障。

当前容器与镜像已重新构建；现有进程需要重启才能采用新 QoS。
临时 `/cable_camera/cable_points_reliable_test` 仅用于诊断，不能作为永久接口。
用户已确认该测试话题连续两次收到 8235 / 8261 点的完整点云；持续帧率尚需远端验证。
