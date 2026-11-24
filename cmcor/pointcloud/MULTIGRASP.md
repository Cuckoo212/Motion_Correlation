# CMCor 多轮扰动会话

GPU 启动参数 `cmcor_mode` 可选 `single`（默认，保持原有单轮行为）或 `multi`。两个模式不能同时运行：它们使用相同的机械臂录制服务和心跳。多轮仍沿用单轮的抓取、扰动、力保护和逐轮 CMCor/SAM3 计算，不会自动控制机械臂。

## 构建和运行

GPU 拉取新代码后，在 `cmcor/pointcloud/docker` 下执行 `sudo docker compose build gpu`。容器镜像中安装了 ROS 包，因此更新 Python 源码和 launch 后也需要重建镜像。随后启动：

```bash
sudo docker compose run --rm --no-deps gpu ros2 launch cable_pointcloud gpu_pc.launch.py cmcor_mode:=multi
```

Mini PC 先按原有方式启动 `mini_pc.launch.py` 发布相机流，再启动 `single_perturbation.launch.py`。确认机器人状态、相机、GPU 和时钟正常后，在 Mini PC 新终端 source ROS 工作区并执行：

```bash
ros2 service call /cable_interaction/start_multigrasp_session std_srvs/srv/Trigger '{}'
```

只在服务返回 `success: true` 后开始第一次抓取。此后手动按原有方式选点、预览并执行，每轮结束后待 CMCor 计算完成，再释放、移动并进行下一轮。会话保持连续录制；同一轮中 X 和 Y 方向结果先按现有单轮逻辑合并。录制过程中可在 `/cable_interaction/cmcor_status` 查看逐轮完成/失败状态。每轮会发布原有红色 raw 和青色 edited 点云。

最后一轮完成并等待其计算完成后，执行：

```bash
ros2 service call /cable_interaction/finish_multigrasp_session std_srvs/srv/Trigger '{}'
ros2 topic echo /cable_interaction/cmcor_status
```

服务成功只表示已接收并开始异步对齐，不代表计算完成；等待 `multi-grasp completed`。不需要结果时，可在没有进行扰动或逐轮计算时调用 `/cable_interaction/abort_multigrasp_session`，数据保留但不做融合。错误和失败结果会写在会话目录下的 `session.json`。不要在会话结束写盘前关闭 GPU 容器。

## 数据与结果

每次会话存于 `cmcor/output/cmcor_sequences/multi-时间-标识/`。`frames.jsonl` 关联原始 PNG RGB、毫米深度、相机参数和 Mini PC 时间戳；`states.jsonl` 记录来自机械臂的时间戳、动作编号、TCP 点、基座坐标系 TCP 位姿和可用的夹爪宽度（不可用为 null）；`phases.jsonl` 记录机械臂阶段事件。各轮的原有数据和掩码位于 `rounds/`，会话结束生成每轮传播掩码、有效区、每像素投票数、有效次数及最终掩码。事件和状态使用其各自原始时间戳，按时间戳关联；并非每帧都有对应的机器人状态样本。

逐轮结果：`/cable_camera/cmcor_points_raw`、`/cable_camera/cmcor_points_edited`；最终多轮结果：`/cable_camera/cmcor_multi_points_raw`（橙）和 `/cable_camera/cmcor_multi_points_edited`（绿）。最终 edited 在最终帧再次请求 SAM3，并与传播结果相交。若任一轮缺少 edited，则不发布多轮 edited；若会话队列溢出、写盘失败、任一轮失败、帧时间不递增或间隔超限，则不发布完整多轮结果。原始结果可能在最终 SAM3 失败前已经发布，必须检查状态和 `session.json`。

对齐在连续帧之间使用现有光流模型，前后向一致且在图像边界内的对应才计入有效区域；它不能保证穿越遮挡后恢复线缆身份。RGB-D 必须保持持续、同一裁剪参数和同步时钟，必须有足够磁盘空间。若计算太慢应等待完成，不要重启。没有进行机械臂遮挡补全，不应将无效区域当成可信背景。

GPU launch 可配置 `max_frame_gap`（秒，默认 0.5）、`flow_consistency_pixels`（默认 1.5）、`min_votes`（每轮合并结果的最低票数，默认 1）和 `session_queue_size`（默认 300）。更大的容忍值不保证跟踪正确；投票阈值增大可能导致遮挡区空缺。
