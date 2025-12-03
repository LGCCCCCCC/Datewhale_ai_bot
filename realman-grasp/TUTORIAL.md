# 教程：基于睿尔曼机械臂的具身智能抓取 Baseline

欢迎来到“AI + 硬件具身智能开源挑战赛”的官方 Baseline 教程！本教程旨在带您深入理解这个端到端的机器人抓取项目，为您在比赛中的创新提供坚实的基础。

---

## 第一章：项目介绍与快速上手

### 1.1 什么是具身智能？
具身智能（Embodied AI）是人工智能研究的前沿领域，旨在让智能体（如机器人）通过物理实体与环境进行交互，从而学习和理解世界。它不仅仅是关于算法，更是关于算法、感知和物理执行的深度融合。本次“AI + 硬件具身智能开源挑战赛”鼓励开发者将AI算法与硬件结合，创造出能够在真实世界中执行任务的智能系统。

### 1.2 项目目标
本项目作为一个官方 Baseline，旨在为您提供一个完整的、端到端的具身智能应用案例。我们将通过 `ManiSkill3` 仿真环境，控制睿尔曼（RealMan）机械臂，实现从“看”（视觉感知）到“抓”（物理执行）的全流程。

### 1.3 成果展示
*(在此处插入一张引人注目的项目截图或GIF动图，例如机械臂成功抓取香蕉的瞬间)*
![Project Demo](./docs/images/demo.gif)

### 1.4 快速开始
在深入代码细节之前，我们强烈建议您先跑通整个流程。请参考项目首页中的 [**快速上手 (Quick Start)**](./TUTORIAL_README.md#️-快速上手-quick-start) 章节，完成环境配置和首次运行。

---

## 第二章：核心技术栈解析

本项目融合了仿真、感知、规划和控制四大模块，以下是关键技术点的解析。

### 2.1 仿真环境：`ManiSkill3` 与 `SAPIEN`
- **SAPIEN**: 一个提供实时物理模拟和逼真渲染的物理引擎。
- **ManiSkill3**: 一个专为具身智能研究设计的、基于SAPIEN的强化学习环境框架。它提供了丰富的机器人、场景和任务，是验证算法的理想平台。
- **为什么选择它?**: 物理真实性高、渲染效果好、社区活跃，非常适合进行Sim2Real（从仿真到现实）的研究。

在我们的项目中，`ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/two_robot_stack_cube_env.py` 文件定义了我们自己的 `ManiSkill` 环境。

```python:16:61:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/two_robot_stack_cube_env.py
# ...
from mani_skill.utils.registration import register_env
# ...

@register_env("TwoRobotStackCube-v1", max_episode_steps=100)
class TwoRobotStackCubeEnv(BaseEnv):
    """
    双臂堆叠立方体环境 - 使用 GEN72-EG2 机器人
    
    任务描述: 两个机器人合作堆叠两个立方体。一个机器人需要拾取蓝色立方体放在目标区域，
    另一个机器人需要拾取绿色立方体并放置在蓝色立方体上方。
    """
    
    # 支持的机器人配置 - 使用新的带相机的 UID
    SUPPORTED_ROBOTS = [(robot_wristcam_uid, robot_wristcam_uid)]
    # ...
```
`@register_env` 装饰器将我们的自定义环境类 `TwoRobotStackCubeEnv` 注册到 `ManiSkill` 框架中，使其可以通过 `gym.make()` 等标准方法被调用。

### 2.2 机器人模型：URDF
- **URDF (Unified Robot Description Format)**: 一种XML格式文件，用于描述机器人的所有物理属性，包括连杆（link）、关节（joint）、视觉模型（visual mesh）和碰撞模型（collision mesh）。
- **在本项目中**: 我们使用睿尔曼官方提供的 `rm_models`。但 `SAPIEN` 引擎在加载包含 `package://` 协议（常用于ROS）的URDF时会遇到问题。因此，我们在 `register_gen72_robot.py` 中编写了一个预处理函数 `get_processed_urdf_path`，它会动态地找到 `mani_skill` 包的路径，替换掉 `package://` 引用，并生成一个 SAPIEN 可以直接加载的 `.processed.urdf` 文件。

```python:23:63:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/register_gen72_robot.py
# --- 动态路径处理 ---

# 1. 自动查找 mani_skill 包的根路径
_mani_skill_root_dir = Path(mani_skill.__path__[0])

# 2. 自动查找项目根目录，以定位原始URDF文件
_this_file = Path(__file__).resolve()
# realman-grasp/ppo.../register... -> realman-grasp
_project_root = _this_file.parent.parent 

def get_processed_urdf_path(original_urdf_path: Path) -> str:
    """
    读取 URDF 文件，将 'package://mani_skill/' 替换为找到的绝对路径，
    并将其保存到一个新的 ".processed.urdf" 文件中。
    返回这个新文件的路径。
    """
    # ... (省略具体实现) ...
    processed_content = content.replace(
        "package://mani_skill/", f"{str(_mani_skill_root_dir)}/"
    )
    # ...
    return str(processed_path)

# 3. 定义原始URDF文件的路径并生成处理后的路径
_original_urdf_path = _project_root / "urdf_01/GEN72-EG2.urdf"
_processed_urdf_path = get_processed_urdf_path(_original_urdf_path)
```
然后，我们使用 `@register_agent` 装饰器将这个自定义机器人“注册”到 `ManiSkill` 框架中，并在类定义中指定使用处理后的URDF路径。

```python:82:89:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/register_gen72_robot.py
@register_agent()
class GEN72EG2Robot(BaseAgent):
    """
    GEN72-EG2 机器人类，集成了7自由度机械臂和EG2夹爪
    """
    uid = "gen72_eg2_robot"
    urdf_path = _processed_urdf_path # 使用处理后的URDF路径
```

### 2.3 “眼睛”——感知模块
感知是具身智能的第一步。本项目在 `perception_pipeline.py` 中构建了一个“视觉模型管线”，协同处理视觉信息。

- **Florence-2**: 微软推出的新一代视觉基础模型。我们使用它的“开放词汇目标检测”能力，仅通过一句文字提示（如 "the banana"），就能在图像中定位出目标物体。

```python:90:102:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/perception_pipeline.py
def detect_object_florence(image_pil, text_prompt):
    """Detects an object using Florence-2 and returns the largest bounding box.

    Args:
        image_pil (PIL.Image): Input RGB image.
        text_prompt (str): Text prompt for the object (e.g., "the banana").

    Returns:
        list or None: Bounding box [x1, y1, x2, y2] or None if not found/error.
    """
    if florence_model is None or florence_processor is None:
        print("Error: Florence-2 model not loaded.")
        return None
```
- **Segment Anything Model (SAM)**: Meta AI 开源的强大的图像分割模型。我们利用 Florence-2 提供的边界框（Bounding Box）作为提示，让 SAM 对目标物体进行像素级的精确分割，生成“掩码（mask）”。

```python:164:190:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/perception_pipeline.py
def segment_object_sam(image_np, bbox):
    # ...
    input_box = np.array(bbox)

    try:
        print("SAM: Setting image...")
        sam_predictor.set_image(image_np)
        print(f"SAM: Predicting mask with bbox: {input_box}")

        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :], # Add batch dimension
            multimask_output=False, # Get single best mask
        )
    # ...
```
- **从2D到3D**: 我们利用腕部深度相机获取的深度图，结合相机内参和 SAM 输出的掩码，筛选出只属于目标物体的点，最终生成该物体的3D点云数据。这一步是连接2D视觉和3D物理世界的桥梁。

```python:205:220:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/perception_pipeline.py
def depth_to_point_cloud(depth, intrinsics, mask=None, rgb=None):
    """
    Converts a depth map to a point cloud, optionally masked and colored.
    Handles depth in millimeters (ManiSkill default) and converts to meters.
    """
    # ...
    # Combine with optional mask
    if mask is not None:
        valid_mask = valid_depth_mask & mask
    else:
        valid_mask = valid_depth_mask
    # ...
    # Calculate x, y coordinates (only for valid points)
    # ...
    points = np.stack([x, y, z], axis=-1)
    points = points[valid_mask] # Select only valid points
    return points, colors
```

### 2.4 “大脑”——决策与规划
获得物体的3D信息后，机器人需要决定“如何抓”。

- **GraspNet**: 一个经典的抓取位姿估计算法，能够从3D点云中预测出多个高质量的六自由度抓取姿态。
- **在本项目中**: 为了降低上手门槛，我们暂时移除了对 `GraspNet` 的强依赖，并使用了一个**占位符**（返回一个虚拟的抓取姿态）。这为您留下了第一个绝佳的创新入口：**您可以尝试集成真实的GraspNet或其他更先进的抓取估计算法！**

你可以在 `grasp_prediction_utils.py` 中看到这个占位符函数：
```python:108:122:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/grasp_prediction_utils.py
def get_grasp_poses_from_graspnet(rgb_image_path, points_np, intrinsics_matrix, checkpoint_path):
    """
    [占位符] 调用 GraspNet 模型进行抓取预测。
    您需要将真实的 GraspNet baseline 推理逻辑替换掉这里的虚拟逻辑。
    """
    print("--- [占位符] 开始 GraspNet 推理 ---")
    # ... (省略参数打印) ...
    grasp_poses_T_Cam_Grasp = []

    # ===========================================================================
    # TODO: 在这里替换为您的真实 GraspNet 推理代码
    # ===========================================================================
    print("警告: GraspNet 推理逻辑未实现，返回虚拟抓取位姿。")
    T_Cam_Grasp_dummy1 = np.identity(4)
    T_Cam_Grasp_dummy1[2, 3] = 0.1 # 假设 Z 轴朝前
    grasp_poses_T_Cam_Grasp.append(T_Cam_Grasp_dummy1)
    # ...
    return grasp_poses_T_Cam_Grasp
```

- **坐标系变换**: 这是具身智能中最关键也最容易出错的一环。GraspNet 输出的抓取位姿是相对于**相机坐标系**的，但机器人执行动作需要的是相对于**机械臂基座坐标系**的目标。因此，我们需要计算 `T_Base_Grasp = T_Base_Cam * T_Cam_Grasp`。这个计算在 `grasp_prediction_utils.py` 中完成。

```python:162:254:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/grasp_prediction_utils.py
# 计算 T_Base_Cam
def get_camera_pose_in_base(agent, camera_link_name="camera_link"):
    # ... (获取基座和相机link的世界位姿)
    # T_World_Base, T_World_Link
    # ...
    # T_Base_Cam = T_Base_World @ T_World_Cam
    return T_Base_Cam

# 计算 T_Base_Grasp
def transform_grasp_to_base(grasp_pose_cam, T_Base_Cam):
    # ...
    T_Base_Grasp = T_Base_Cam @ grasp_pose_cam
    return T_Base_Grasp
```
### 2.5 “手臂”——控制与执行
最后一步是将规划好的动作执行出来。

- **逆运动学 (Inverse Kinematics, IK)**: 我们知道末端执行器（夹爪）想要到达的目标姿态，但需要反向求解出每个关节应该旋转的角度。这是一个复杂的数学问题。
- **PyBullet**: 我们使用 `PyBullet` 物理引擎内置的高效IK求解器来完成这个计算。`main_grasp_workflow.py` 中的 `compute_ik_with_pybullet` 函数封装了整个流程。

```python:59:71:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/main_grasp_workflow.py
def compute_ik_with_pybullet(
    urdf_path: str,         # URDF 文件路径 (建议先用简化版 GEN72.urdf)
    ee_link_name: str,      # 末端执行器 Link 名称 (来自 SAPIEN)
    target_pose_base: sapien.Pose, # 目标位姿 (相对于基座)
    current_sapien_qpos: torch.Tensor, # SAPIEN 当前 qpos [1, N_active]
    all_sapien_joint_names: list[str], # SAPIEN 活动关节名称列表
    device: torch.device,
    dtype: torch.dtype
):
    """使用 PyBullet 计算逆运动学"""
    print("--- 开始执行 IK (PyBullet) --- ")
```
这个函数的核心是调用 `p.calculateInverseKinematics`。它需要加载一个简化的URDF（无碰撞体，计算更快），并正确映射SAPIEN和PyBullet之间的关节索引。

- **控制器**: 求解出目标关节角度后，我们将这些角度作为指令，发送给 `ManiSkill` 中的 `pd_joint_pos` 关节位置控制器，驱动仿真世界中的机械臂完成动作。在主流程中，IK求解出的7个关节角度会和一个夹爪动作合并成一个8D的动作向量，然后包装成字典传给 `env.step()`。

```python:919:956:ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/main_grasp_workflow.py
                            # --- 修改: 使用 env.step(action) 并构造 8D 动作 --- 
                            # ... (获取当前左右臂qpos) ...
                            try:
                                # --- 构造左臂 8D 动作 --- 
                                target_qpos_left_7d = ik_qpos_7d.squeeze(0) # Shape [7]
                                # ... (获取当前夹爪值) ...
                                current_gripper_left = qpos_left_current[0, gripper_joint_index:gripper_joint_index+1] # Shape [1]
                                action_left_8d = torch.cat((target_qpos_left_7d, current_gripper_left), dim=0).to(device=device, dtype=dtype) # Shape [8]

                                # --- 构造右臂 8D 动作 (保持当前姿态) ---
                                # ...
                                action_right_8d = qpos_right_current.squeeze(0)[:num_controlled_joints].to(device=device, dtype=dtype) # Shape [8]

                                # --- 获取 Action Keys --- 
                                action_space_keys = list(env.action_space.spaces.keys())
                                # ...
                                left_agent_key = action_space_keys[0]
                                right_agent_key = action_space_keys[1]

                                # --- 构建最终 Action 字典 --- 
                                action = {
                                    left_agent_key: action_left_8d,
                                    right_agent_key: action_right_8d
                                }
                            # ...
                            # 循环调用 env.step(action)
                            num_steps = 50
                            if action is not None:
                                for step_idx in range(num_steps):
                                    obs, reward, terminated, truncated, info = env.step(action)
                                    if env.render_mode == "human":
                                            env.render()
```

---

## 第三章：代码结构导览

```
realman-grasp/
├── ppo_my_ok_demo15_all_pipe_ok_only_grasp_to_do/  # 核心工作目录
│   ├── main_grasp_workflow.py         # 主程序入口，串联所有流程
│   ├── two_robot_stack_cube_env.py    # ManiSkill环境定义
│   ├── register_gen72_robot.py        # 自定义机器人注册脚本
│   ├── perception_pipeline.py         # 视觉感知管线 (Florence-2, SAM)
│   └── grasp_prediction_utils.py      # 抓取位姿估计相关工具 (GraspNet占位符)
├── urdf_01/                             # 存放机器人URDF文件
├── my_urdf/                             # 存放机器人模型资源 (需从Gitee克隆)
├── checkpoints/                         # 存放模型权重 (需手动下载)
└── TUTORIAL.md                          # 本教程文件
```

**`main_grasp_workflow.py`** 是整个项目的“指挥中心”。它的核心逻辑在 `test_set_pose_and_capture()` 函数中，可以分为以下几个阶段：
1.  **初始化与观察**:
    *   创建并重置 `TwoRobotStackCubeEnv` 环境。
    *   机械臂移动到一个预设的观察姿态。
    *   调用腕部相机拍摄第一张RGB和深度图。
2.  **（已禁用）初步感知**:
    *   （这部分代码当前被禁用，直接使用一个固定的世界坐标点代替）理论上，这里会调用 `perception_pipeline` 来找到物体在世界坐标系中的大概位置。
3.  **接近目标（Pre-Grasp）**:
    *   计算出物体上方的一个“预抓取”位姿。
    *   调用 `compute_ik_with_pybullet` 计算到达该位姿所需的关节角度。
    *   通过 `env.step()` 控制机械臂移动到预抓取位置。
    *   拍摄第二张（也是用于抓取决策的）RGB和深度图。
4.  **抓取决策**:
    *   使用第二张深度图和相机内参生成点云 (`depth_to_point_cloud`)。
    *   将点云和RGB图输入 `get_grasp_poses_from_graspnet`（占位符），得到一个相机坐标系下的抓取位姿 `T_Cam_Grasp`。
    *   进行坐标变换，得到基座坐标系下的抓取位姿 `T_Base_Grasp`。
5.  **执行抓取**:
    *   再次调用 `compute_ik_with_pybullet`，这次的目标是 `T_Base_Grasp`。
    *   通过 `env.step()` 控制机械臂移动到最终的抓取位置。
6.  **结束**: 保持仿真窗口，直到用户手动关闭。

---

## 第四章：如何改进与参与比赛

本项目为您提供了一个坚实的起点，但还有巨大的创新空间。以下是一些建议，可以帮助您在比赛中脱颖而出：

### 4.1 [高优先级] 集成真实的抓取位姿估计算法
**你的首要任务**：将 `grasp_prediction_utils.py` 中的占位符函数 `get_grasp_poses_from_graspnet` 替换为真实的模型推理代码。
- **步骤**:
    1.  克隆 [GraspNet](https://github.com/graspnet/graspnet-baseline) 官方仓库或寻找其他SOTA（State-of-the-Art）的抓取检测模型。
    2.  根据其文档安装依赖，并下载预训练模型。
    3.  在 `get_grasp_poses_from_graspnet` 函数中，加载模型，并根据其API对输入的点云和RGB图像进行预处理。
    4.  执行模型推理，并对输出进行后处理，使其符合函数要求的返回格式：一个包含多个4x4 `T_Cam_Grasp` 矩阵的列表。

### 4.2 优化抓取策略
GraspNet通常会返回多个候选抓取姿态。当前代码只是简单地选择了第一个 (`predicted_grasps_cam_frame[0]`)。
- **创新点**:
    *   **碰撞检测**: 在选择抓取位姿前，模拟一个虚拟的夹爪，检查它在目标位姿是否会与桌面或其他物体发生碰撞。
    *   **可达性分析**: 检查机械臂是否能够无障碍地到达某个抓取位姿。
    *   **质量排序**: 根据GraspNet输出的置信度分数或其他指标（如夹爪张开宽度）对抓取进行排序。

### 4.3 引入任务规划
当前机器人只能执行单一的抓取指令。
- **创新点**:
    *   **结合大语言模型(LLM)**: 让机器人能理解更复杂的自然语言指令，如“把桌子上的香蕉放到红碗里”。你需要：
        1.  用LLM解析指令，识别出操作对象（香蕉）、目标位置（红碗）。
        2.  扩展感知模块，使其能定位所有相关物体。
        3.  生成一个动作序列：`[移动到香蕉上方 -> 抓取香蕉 -> 移动到红碗上方 -> 释放香蕉]`。
    *   **状态机/行为树**: 设计一个更传统的任务规划器来处理复杂任务流。

### 4.4 实现Sim2Real
这是最具挑战也最有价值的一步。尝试将您在仿真中验证通过的算法，部署到真实的睿尔曼机械臂上。
- **关键挑战**:
    *   **相机标定**: 精确获取真实相机的内参和它相对于机械臂末端的（手眼）外参。
    *   **执行误差**: 真实世界的电机控制存在误差，需要考虑鲁棒性。
    *   **感知差异**: 仿真中的RGBD数据是完美的，真实世界则充满噪声和光照变化。

祝您在比赛中取得成功！
