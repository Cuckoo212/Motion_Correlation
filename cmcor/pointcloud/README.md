# cable_pointcloud：ROS 2 Humble 实时 Cable 点云

ROS 包和 Python 模块统一命名为 `cable_pointcloud`，旧 `pointcloud` Python 模块、
独立直连入口及 TCP 点云回传代码已删除。外层 `pointcloud/` 仅作为包源码目录保留，
以保持权重和挂载路径稳定。GPU 节点全部在一个 Humble + SAM3 Docker 容器内运行。

## 1. 五节点流程

```text
Mini PC                              GPU PC：Humble + SAM3 Docker

① 相机采集发布节点 ── RGB-D ───────→ ② RGB-D 接收节点
                                          ↓ 最新完整帧对
                                    ③ SAM3 点云计算节点
                                          ↓ 点云结果
⑤ 点云接收显示节点 ←─ PointCloud2 ── ④ 点云发布节点
```

| 节点 | 运行位置 | 职责 |
| --- | --- | --- |
| ① `rgbd_capture_publisher` | Mini PC | 采集 RealSense、对齐彩色和深度、发布图像及相机信息 |
| ② `rgbd_receiver` | GPU Docker | 订阅图像，按时间戳配对，缓存最新完整 RGB-D 帧 |
| ③ `sam3_pointcloud` | GPU Docker | SAM3 分割、深度反投影、下采样、保存图像和 PLY，输出点云结果 |
| ④ `pointcloud_publisher` | GPU Docker | 发布标准 `sensor_msgs/msg/PointCloud2`，通过 ROS 2 回传 |
| ⑤ `pointcloud_viewer` | Mini PC | 订阅点云，在 Mini PC 本地显示 Matplotlib 实时窗口 |

采集和图像发布放在同一个节点，因为它们共用相机数据，拆开会增加一次数据传递。
GPU 端保留接收、计算、发布三个节点，让 ROS 通信和较慢的 GPU 推理独立运行；
显示或点云回传中断时，推理仍可继续保存结果。RGB-D 输入断流超过超时则保存元数据并退出。

GPU 三个节点放在同一个 Humble + SAM3 容器。接收和发布节点使用 Humble 系统 Python；
计算节点使用可访问 Humble `rclpy` 的 SAM3 Python 环境。接收至计算使用有界的最新帧缓存和
本地 Unix socket；点云通过有界 ROS 话题队列传给发布节点。跨机器统一使用 ROS 2 DDS，
替代旧 TCP 点云回传方案。

## 2. 代码结构

```text
pointcloud/                         # ROS 包源码根目录
├── package.xml                     # ament_python 包声明
├── setup.py / setup.cfg             # 安装和 ros2 run 入口
├── resource/cable_pointcloud        # ament 包索引
├── cable_pointcloud/                # 唯一 Python 模块
│   ├── __init__.py
│   ├── rgbd_capture_publisher.py     # ① 相机采集发布
│   ├── rgbd_receiver.py              # ② 接收、配对、socket 服务
│   ├── ros_nodes.py                  # ③④⑤ ROS 入口和 PointCloud2 编解码
│   ├── inference.py                  # SAM3、点云保存和频率统计
│   ├── inference_args.py             # 推理参数
│   ├── sam3_segmenter.py             # SAM3 封装
│   ├── rgbd_transport.py             # 最新帧缓存和本地协议
│   ├── geometry.py                   # 反投影、下采样、PLY
│   └── runtime.py                    # JSON 和 Matplotlib 显示
├── launch/
│   ├── mini_pc.launch.py             # ①⑤
│   └── gpu_pc.launch.py              # ②③④
├── docker/
│   ├── Dockerfile.sam3               # Humble + SAM3 单镜像
│   ├── compose.yaml                  # 唯一 gpu 服务
│   ├── ros-entrypoint.sh             # 自动加载 ROS 环境
│   ├── build-sam3.sh                 # 安装 GPU 容器运行时、构建、自检
│   └── runtime/                      # socket 目录，不提交 Git
├── requirements-realtime.txt
├── tests/
└── sam3_project/                     # 独立 SAM3 源码和权重
    ├── COLCON_IGNORE                # 不作为 ROS 包构建
    ├── sam3/
    └── checkpoints/sam3.pt           # 权重不提交 Git
```

## 3. GPU PC 构建：在宿主机执行

先停止旧推理和采集。旧 receiver/inference 容器若还在运行，用 `sudo docker ps`
找到容器名，再用 `sudo docker stop <容器名>` 停止，避免 socket 被占用。

```bash
cd /home/flexcycle/qihang/Motion_Correlation/cmcor/pointcloud/docker
mkdir -p runtime
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
source ~/cable_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch cable_pointcloud mini_pc.launch.py publish_fps:=30 --width 640 --height 480  --crop 100 140 400 300
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
```
创建容器时，入口脚本 `ros-entrypoint.sh` 自动加载 Humble 和
`/opt/cmcor/install/setup.bash`，无需手动 source。以下命令在容器内执行：

```bash
ros2 pkg executables cable_pointcloud
ros2 launch cable_pointcloud gpu_pc.launch.py
```

一次启动接收、SAM3 计算和点云发布；GPU 无需桌面窗口，在 Mini PC 查看点云。
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
图像、mask、元数据和 PLY 保存在 GPU 宿主机：

```text
cmcor/datasets/realtime_cable_pc/<序列名>/
cmcor/output/realtime_cable_pc/<序列名>/
```

先在 GPU launch 终端按 Ctrl-C，等待元数据保存完成，再停止 Mini PC launch。
若进入 bash 启动，最后输入 exit 退出容器。配对计数不增长时检查发布节点、域 ID、
RMW、网络及防火墙。异常退出遗留 socket 时，先确认旧进程停止再清理。

## 7. 测试

在包源码根目录加载 ROS 环境后执行：

```bash
python3 -m unittest discover -s tests -v
```

完整测试依赖 SAM3 运行依赖及 rclpy；真实相机、Humble 镜像和跨机通信需在目标环境验证。
