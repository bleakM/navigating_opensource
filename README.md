# navigating_opensource · 小北 G1 Core

Unitree G1 在 MuJoCo 中自主探索、三维建图、重定位、导航，以及通过声音寻找并用人脸确认呼叫者的核心代码。仅包含算法接口、ROS 2 配置、仿真适配和命令行工具。

## 算法链路

- **MuJoCo 3.3.6 + Grove-G1**：真实物理步进、LiDAR、IMU、RGB-D；沿用预训练 AGILE 腿部策略，使用现有上肢轨迹控制器实现有限摆臂。
- **FAST-LIO2 → RTAB-Map**：前者发布局部惯性里程计，后者负责全局地图、回环、优化及保存地图重定位。RTAB-Map 是 `map → odom` 的唯一发布方。
- **Canopy → Nav2**：从已观测地图选择探索视点和房间，使用官方路径平滑、速度平滑及碰撞保护执行。机器人半径 0.45 m、膨胀半径 1.2 m，期望速度 0.60 m/s，实际速度受控制器及场景约束。
- **四通道波形 → sherpa-onnx 唤醒 → CAMPPlus 声纹 → SRP-PHAT**：识别“小北小北”，在测量时刻通过 TF 转换方位；保留多房间候选与衰减的位置记忆。
- **RGB-D → YuNet / SFace → 安全接近**：只使用脸部特征匹配呼叫者。检测丢失先取消运动再重获；到达需新鲜人脸、合适停距和实际静止。活动任务期间其他呼叫不能抢占。

仿真器仅在生成传感器和声波时读取演员及机器人真值；音频感知端只收到波形，不收到声源位置或人物 ID。导航、SLAM 和人物确认均使用估计结果。声场采用镜像声源、门洞等效路径和透墙衰减，是工程近似，未做真实建筑声学标定。

## 目录

```text
core/src/           声学、身份、任务调度、探索监督和摆臂
core/launch/        最终版传感器、SLAM、导航和完整会话
core/config/        RTAB-Map、Nav2、Canopy、行为树、相机参数
core/native/        演员及传感器适配，无网页渲染快照
interaction/src/    声源估计、观测地图搜索、安全接近几何
interaction/ros/    ROS 2 消息定义
g1/                 固定上游版本、必要补丁、基线配置
examples/           展览馆场景描述及会话示例
models/manifest.json 模型官方来源、许可证和 SHA-256
scripts/            环境、编译、CLI、地图保存、模型检查
core/tests/         小型安全与调度回归
```

权重、人物网格、私人录音、声纹/人脸特征、地图数据库、rosbag 和历史报告均由使用者在本地管理，不进入 Git。无网页 App、资源商店、网页后端或视频录制功能。

## 环境与启动

已验证平台：x86-64 Ubuntu 22.04 宿主，通过 Bubblewrap 隔离 Ubuntu 24.04 / ROS 2 Jazzy、Python 3.12；MuJoCo 3.3.6。宿主 Humble 与隔离环境 Jazzy 不混用。需要 NVIDIA/OpenGL 或 headless Xvfb，保持现有驱动即可。

### 复用已准备的依赖

`PREPARED_ROOT` 应包含 `.runtime/noble`、`.runtime/workbench` 或 `.runtime/python`、`.runtime/interaction`、`g1_ws/install`、`interaction_ws/install` 和固定版本 Grove 源码。可复用原工程已验证依赖，状态和身份库单独存放于本仓库 `.runtime/session/`，不会自动导入原身份绑定。

```bash
./scripts/setup --reuse /path/to/PREPARED_ROOT
./scripts/xiaobei run
```

默认从空地图运行 216 m² 展览馆、无人物。需要预备 Grove 家具资产，来源许可见上游资产清单。原生 MuJoCo 窗口用于查看仿真，可另开 RViz2：

```bash
G1_GPU=1 ./scripts/g1-runtime rviz2 -d /workspace/g1/config/mapping.rviz
./scripts/xiaobei status
```

新环境的依赖脚本为 `scripts/bootstrap`（隔离 Jazzy）和 `scripts/g1-bootstrap`（固定 Grove 依赖和补丁）。家具资源须按 Grove 上游 `scripts/setup-world-assets.sh` 的说明准备，不能只运行安装命令就假定资源已存在。再用 Python 3.12 的 `--system-site-packages` 虚拟环境安装 `requirements.txt` 到 `.runtime/python`，构建消息：

```bash
./scripts/g1-runtime colcon build --base-paths /workspace/interaction/ros --build-base /workspace/.runtime/interaction-build --install-base /workspace/interaction_ws/install
./scripts/g1-runtime python3 /workspace/scripts/fetch-models.py
./scripts/setup
```

此源码拆分版已在现有隔离依赖上验证；全新机器从零安装尚未完整复验。不要在宿主执行这些项目依赖的 APT 或混用宿主 Python。

### 地图保存、加载及导航

探索树结束后自动保存数据库、二维地图、点云及图节点，再进行实际地图匹配。树成功只代表其视点调度结束，不等于独立覆盖率验收。

```bash
./scripts/xiaobei status
# Ctrl-C 正常结束运行；用上一步输出的 saved_database 替换文件名
./scripts/xiaobei run --mode saved --database maps/your-saved-map.db
./scripts/xiaobei goal 2.0 1.0
```

保存图不直接写入；启动时复制成会话数据库。加载同一场景与起点；如手动给出初始化猜测，可用 `--initial-pose 'x y z roll pitch yaw'`，仍需真实 RTAB-Map 匹配通过后才启用导航。不能手动移动运行中的机器人。

### 声纹与脸部绑定、呼叫

人物资源放在 `assets/avatars/actor_a/`，包含经过正确转换的 `person-raised.obj` 和 `person.png`。Rocketbox 原始 FBX 需自行转换并保留 Microsoft 上游 MIT 许可；这里不提供资源下载商店或重分发人物包。

```bash
./scripts/xiaobei enroll --identity actor_a --face .local/face.jpg --voice .local/call.wav
```

图片必须只检测到一张清晰脸；绑定音频须清楚说“小北小北”。将 `examples/actors.example.json` 中对应演员复制到你本地的会话 `actors` 列表，最多四人；人物 ID 应与对应网格目录及身份库一致。用 `--manifest .local/session.json` 启动。人物无碰撞体积，并从建图传感器中排除；仅交互相机能看到人物。角色在视野中朝向机器人，不代表识别算法获得身份答案。

```bash
./scripts/xiaobei play --slot 0
./scripts/xiaobei play --slot 0 --wav .local/new-call.wav
./scripts/xiaobei cancel
```

`play` 先检查入口声纹，随后在对应演员头部生成四通道传播音频；机器人重新从波形识别。输出 `.runtime/session/audio-monitor/*.wav` 是机器人接收端通道 0，可用播放器监听。地图未就绪时不响应呼叫。`cancel` 是取消请求，需通过 `status` 确认停车，不能假设提交请求已等于停车。

## 验证范围

来源版本 `10e8d16e2ec6a6a30dacabd18d9c4e871a741f46` 的最终场景完成过建图、保存/重启重定位及指定摆位三人连续寻人；并不代表任意人物位置均成功。源码来源与逐文件哈希见 `SOURCE_PROVENANCE.json`。

本仓库的拆分验证见 `VALIDATION.md`。必要回归可运行：

```bash
./scripts/core-python -m unittest discover -s core/tests -v
./scripts/core-python /workspace/scripts/fetch-models.py --check
```

现实机器人、真实室内声场、任意地图和任意真人身份均未完成验收。预训练步态未重新训练，安全恢复存在次数和时间上限。不要把仿真成功解释为实机安全认证。

## 许可

独立适配代码与配置采用 Apache-2.0。第三方代码、补丁、模型、人物和家具遵守各自许可，详见 `THIRD_PARTY_NOTICES.md`。FAST-LIO 的 GPL、Grove/Unitree 的 BSD 以及步态权重的独立条款不被根许可证覆盖。
