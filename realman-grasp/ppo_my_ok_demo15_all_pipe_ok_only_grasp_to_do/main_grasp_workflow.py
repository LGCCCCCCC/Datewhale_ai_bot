#!/usr/bin/env python3
"""
测试双臂堆叠立方体环境 - 修改为设置姿态、拍照并保持
"""


import numpy as np
import gymnasium as gym
import time
import sys
import os
import cv2 # 添加cv2库
import open3d as o3d # 导入 open3d
import torch # 确保导入 torch
import sapien # 导入 sapien 用于创建标记点
from mani_skill.utils import common # 导入 common 以使用四元数工具
from transforms3d.quaternions import quat2mat # 用于四元数转旋转矩阵
import transforms3d.quaternions as tq # 新增导入
import transforms3d.euler # 新增导入 euler
import transforms3d.axangles as tax # 新增: 用于轴角转换
import ikpy.chain # 新增: 导入 ikpy
import ikpy.utils.geometry # 新增: ikpy 工具
import pybullet as p
import pybullet_data

# --- 新增: 导入抓取预测工具 --- 
from grasp_prediction_utils import (
    depth_to_point_cloud,
    get_grasp_poses_from_graspnet, # 导入占位符函数
    get_camera_pose_in_base,
    transform_grasp_to_base,
    matrix_to_sapien_pose,
    pose_to_transformation_matrix, # 如果本文件还需要它
    visualize_grasps, # 新增: 可视化工具
    graspgroup_to_matrices # 新增: GraspGroup 转换工具
)
# --- 新增结束 ---

# 尝试导入 GraspGroup（用于类型检查）
try:
    from graspnetAPI import GraspGroup
    GRASPNET_AVAILABLE = True
except ImportError:
    GRASPNET_AVAILABLE = False
    GraspGroup = None

# --- 新增: 导入感知模块
# 直接导入，不捕获异常，以便查看具体错误
from perception_pipeline import run_perception
print(f"成功导入 run_perception: {run_perception}")
# --- 新增结束 ---

# --- 新增: 路径管理 ---
from pathlib import Path
_this_file = Path(__file__).resolve()
_this_file_dir = _this_file.parent
_project_root = _this_file_dir.parent # 修正: .parent.parent 错误, .parent 才是正确的项目根目录

# --- 新增结束 ---

# --- 新增: 平滑移动辅助函数 ---
def smooth_move_to_qpos(env, target_qpos_7d, gripper_val, num_steps=100, render=True):
    """
    平滑移动到目标关节位置 (线性插值)。
    
    Args:
        env: 仿真环境。
        target_qpos_7d (torch.Tensor): 目标 7D 手臂关节位置 (Shape: [7] or [1, 7])。
        gripper_val (float): 夹爪目标值 (保持不变或改变)。
        num_steps (int): 插值步数，越多越平滑。
        render (bool): 是否渲染。
    """
    device = env.device
    agent_left = env.agent.agents[0]
    agent_right = env.agent.agents[1] # 右臂保持不动
    
    # 1. 获取当前 qpos
    current_qpos_left = agent_left.robot.get_qpos() # [1, 13]
    current_qpos_right = agent_right.robot.get_qpos() # [1, 13]
    
    # 提取左臂当前 7D 姿态
    current_qpos_left_7d = current_qpos_left[0, :7]
    
    # 确保 target_qpos_7d 是 1D tensor
    if target_qpos_7d.ndim > 1: target_qpos_7d = target_qpos_7d.squeeze(0)
    
    # 2. 生成插值序列
    # shape: [num_steps, 7]
    traj_qpos = torch.zeros((num_steps, 7), device=device, dtype=current_qpos_left.dtype)
    for i in range(7):
        traj_qpos[:, i] = torch.linspace(current_qpos_left_7d[i], target_qpos_7d[i], num_steps)
        
    # 3. 执行插值移动
    gripper_tensor = torch.tensor([gripper_val], device=device, dtype=current_qpos_left.dtype)
    
    # Action Keys
    action_space_keys = list(env.action_space.spaces.keys())
    left_key = action_space_keys[0]
    right_key = action_space_keys[1]
    
    # 右臂保持当前位置 (截取前8个受控关节)
    action_right_8d = current_qpos_right.squeeze(0)[:8]

    for step in range(num_steps):
        # 构造左臂动作: 当前步的 7D 手臂 + 固定的夹爪
        action_left_8d = torch.cat((traj_qpos[step], gripper_tensor), dim=0)
        
        action = {
            left_key: action_left_8d,
            right_key: action_right_8d
        }
        
        try:
            env.step(action)
            if render and env.render_mode == "human": env.render()
        except Exception as e:
            print(f"平滑移动 step {step} 出错: {e}")
            break

# --- 新增: 辅助函数将 (p, q) 转换为 4x4 变换矩阵 ---
def pose_to_transformation_matrix(position, quaternion_wxyz):
    """将 (位置, 四元数 WXYZ) 转换为 4x4 变换矩阵"""
    # ikpy 通常使用 XYZW, transforms3d 使用 WXYZ
    mat = tq.quat2mat(quaternion_wxyz)
    transform = np.identity(4)
    transform[:3, :3] = mat
    transform[:3, 3] = position
    return transform
# --- 新增结束 ---

# --- 新增: 使用 IKPy 的 IK 函数 ---
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
    physics_client = None # 用于确保断开连接
    try:
        # --- 1. 连接到 PyBullet (非图形化) ---
        # 使用 DIRECT 模式，我们不需要图形界面，只进行计算
        physics_client = p.connect(p.DIRECT)
        if physics_client < 0:
             print("错误: 连接 PyBullet 失败!")
             return None, False
        
        # (可选) 添加数据路径，以防 URDF 引用标准资源
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(f"尝试加载 URDF: {urdf_path}")
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # 禁用渲染以加速

        # --- 2. 加载 URDF ---
        # 加载 URDF，禁用实时模拟，设置一个固定的时间步（虽然DIRECT模式下可能不严格需要）
        p.setTimeStep(1./240.)
        # 使用 USE_INERTIA_FROM_FILE 可能更准确，如果 URDF 中惯性正确的话
        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        print(f"PyBullet 加载 URDF 成功, Robot ID: {robot_id}")

        # --- 3. 构建 SAPIEN 名称到 PyBullet 索引和信息的映射 ---
        num_joints_pb = p.getNumJoints(robot_id)
        print(f"PyBullet 找到的总关节数: {num_joints_pb}")
        
        sapien_name_to_pb_index = {}
        pb_joint_indices_active = [] # 只存储与 SAPIEN 活动关节对应的 PB 索引
        pb_lower_limits = []
        pb_upper_limits = []
        pb_joint_ranges = []
        pb_rest_poses = []
        ee_link_pb_index = -1 # 初始化末端执行器 Link 的 PyBullet 索引
        
        # 标准化 SAPIEN 名称以进行可靠比较
        normalized_sapien_active_names_set = {name.strip() for name in all_sapien_joint_names}

        print("构建关节映射 (SAPIEN -> PyBullet)...")
        for i in range(num_joints_pb):
            joint_info = p.getJointInfo(robot_id, i)
            pb_joint_index = joint_info[0]
            pb_joint_name_bytes = joint_info[1]
            pb_joint_name = pb_joint_name_bytes.decode('utf-8').strip()
            pb_joint_type = joint_info[2]
            pb_link_name_bytes = joint_info[12]
            pb_link_name = pb_link_name_bytes.decode('utf-8').strip()
            pb_ll = joint_info[8]
            pb_ul = joint_info[9]

            print(f"  PB Joint Index {pb_joint_index}: Name='{pb_joint_name}', Type={pb_joint_type}, Link='{pb_link_name}', Limits=({pb_ll:.4f}, {pb_ul:.4f})", end="")

            # 检查这个关节是否是我们关心的活动关节 (来自 SAPIEN)
            # 并且它必须是可动的 (非固定)
            if pb_joint_name in normalized_sapien_active_names_set and pb_joint_type != p.JOINT_FIXED:
                sapien_name_to_pb_index[pb_joint_name] = pb_joint_index
                pb_joint_indices_active.append(pb_joint_index)
                pb_lower_limits.append(pb_ll)
                pb_upper_limits.append(pb_ul)
                pb_joint_ranges.append(pb_ul - pb_ll)
                # 使用中间位置作为 rest pose
                pb_rest_poses.append((pb_ll + pb_ul) / 2.0)
                print(" -> 活动关节 (匹配 SAPIEN)")
            else:
                 print(" -> 非活动或固定关节")

            # 检查这个 joint 连接的 link 是否是我们的目标 EE link
            if pb_link_name == ee_link_name:
                ee_link_pb_index = pb_joint_index # PyBullet IK 需要的是 Joint Index
                print(f"    -> 找到末端执行器 Link '{ee_link_name}' 对应的 PyBullet Joint Index: {ee_link_pb_index}")

        # 验证是否找到了 EE link 索引
        if ee_link_pb_index == -1:
            print(f"错误: 未能在 PyBullet 中找到名为 '{ee_link_name}' 的末端执行器 Link 对应的 Joint Index。可用 Link 名称:")
            for i in range(num_joints_pb): print(f"  - {p.getJointInfo(robot_id, i)[12].decode('utf-8')}")
            return None, False

        # 验证是否找到了所有 SAPIEN 活动关节
        if len(pb_joint_indices_active) != len(all_sapien_joint_names):
             print(f"警告: PyBullet 中找到的可动关节数 ({len(pb_joint_indices_active)}) 与 SAPIEN 活动关节数 ({len(all_sapien_joint_names)}) 不匹配！")
             # 打印哪些缺失
             pb_found_names = set(sapien_name_to_pb_index.keys())
             missing_in_pb = normalized_sapien_active_names_set - pb_found_names
             if missing_in_pb: print(f"  SAPIEN 关节未在 PyBullet 中找到或非可动: {missing_in_pb}")
             # 可以在这里决定是否返回失败
             # return None, False
             
        print(f"SAPIEN 名称到 PyBullet 索引的映射: {sapien_name_to_pb_index}")
        print(f"用于 IK 的 PyBullet 活动关节索引列表: {pb_joint_indices_active}")


        # --- 4. 准备 IK 输入 ---
        target_pos = target_pose_base.p # 期望位置 [x, y, z]
        # PyBullet 需要 xyzw 格式的四元数
        target_orn_wxyz = target_pose_base.q
        target_orn_xyzw = [target_orn_wxyz[1], target_orn_wxyz[2], target_orn_wxyz[3], target_orn_wxyz[0]]
        
        print(f"目标位置 (基座标系): {target_pos}")
        print(f"目标姿态 (基座标系 quat xyzw): {target_orn_xyzw}")

        # 将 SAPIEN 当前 qpos 映射到 PyBullet 的活动关节顺序
        current_sapien_qpos_np = current_sapien_qpos.squeeze(0).cpu().numpy()
        sapien_idx_map = {name: idx for idx, name in enumerate(all_sapien_joint_names)}
        
        # IMPORTANT: calculateInverseKinematics expects currentPosition for *all* movable joints
        # it will use based on the indices passed implicitly via lowerLimits etc.
        # We need to provide the current positions in the order defined by pb_joint_indices_active
        print("映射 SAPIEN qpos 到 PyBullet 初始猜测...")
        num_active_pb_joints = len(pb_joint_indices_active)
        if len(pb_lower_limits) != num_active_pb_joints: print("错误: 限制数量与活动关节数不匹配!") # Sanity check
        
        initial_guess_pb = [0.0] * num_active_pb_joints # Initialize
        map_success = True
        for i, pb_idx in enumerate(pb_joint_indices_active):
             joint_info = p.getJointInfo(robot_id, pb_idx)
             pb_joint_name = joint_info[1].decode('utf-8').strip()
             if pb_joint_name in sapien_idx_map:
                  sapien_idx = sapien_idx_map[pb_joint_name]
                  try:
                      initial_guess_pb[i] = current_sapien_qpos_np[sapien_idx]
                      # print(f"  Mapped PB idx {pb_idx} ('{pb_joint_name}') from SAPIEN idx {sapien_idx} = {initial_guess_pb[i]:.4f}")
                  except IndexError:
                      print(f"错误: SAPIEN 索引 {sapien_idx} 超出 qpos 范围 (长度 {len(current_sapien_qpos_np)}) for joint '{pb_joint_name}'")
                      map_success = False
                      break
             else:
                  print(f"错误: PyBullet 活动关节 '{pb_joint_name}' (idx {pb_idx}) 未在 SAPIEN 索引映射中找到!")
                  map_success = False
                  break
        if not map_success:
             print("映射 SAPIEN qpos 到 PyBullet 失败。")
             return None, False
        print(f"PyBullet 初始关节猜测 (len {len(initial_guess_pb)}): {np.round(initial_guess_pb, 4)}")

        # --- 5. 调用 PyBullet IK ---
        print("调用 PyBullet calculateInverseKinematics (带限制，无初始猜测)...") # Modified print
        # solver=0 uses the default DLS solver. Can try p.IK_SDLS for Selective Damped Least Squares
        # maxNumIterations and residualThreshold control convergence
        # --- 修改: 保留关节空间参数，移除 currentPosition 和优化参数 --- 
        ik_solution_pb = p.calculateInverseKinematics(
            robot_id,
            ee_link_pb_index,
            target_pos,
            target_orn_xyzw,
            lowerLimits=pb_lower_limits,    # 保留
            upperLimits=pb_upper_limits,    # 保留
            jointRanges=pb_joint_ranges,    # 保留
            restPoses=pb_rest_poses,        # 保留
            # currentPosition=initial_guess_pb, # 移除
            # solver=0,                     # 移除
            # maxNumIterations=200,         # 移除
            # residualThreshold=.001          # 移除
        )
        # --- 修改结束 ---
        
        # 检查返回结果长度是否与活动关节数匹配
        if ik_solution_pb is None or len(ik_solution_pb) != num_active_pb_joints:
             print(f"错误: PyBullet IK 计算失败或返回结果长度 ({len(ik_solution_pb) if ik_solution_pb else 'None'}) 与活动关节数 ({num_active_pb_joints}) 不匹配。")
             # 尝试不带 currentPosition 再算一次 (现在这个调用和上面一样了)
             print("再次调用 IK (参数与第一次相同)...") # Modified print
             # --- 修改: 第二次尝试也移除 currentPosition 等 --- 
             ik_solution_pb = p.calculateInverseKinematics(
                 robot_id, ee_link_pb_index, target_pos, target_orn_xyzw,
                 lowerLimits=pb_lower_limits, # 保留
                 upperLimits=pb_upper_limits, # 保留
                 jointRanges=pb_joint_ranges, # 保留
                 restPoses=pb_rest_poses    # 保留
                 # currentPosition 移除
                 # solver, maxNumIterations, residualThreshold 移除
             )
             # --- 修改结束 ---
             if ik_solution_pb is None or len(ik_solution_pb) != num_active_pb_joints:
                  print("再次尝试 IK 失败。")
                  return None, False
             else:
                  print("不使用 currentPosition 的 IK 成功。")
                     
        print(f"PyBullet IK 求解完成。解 (len {len(ik_solution_pb)}): {np.round(ik_solution_pb, 4)}")

        # --- 6. 将结果映射回 SAPIEN 格式 (返回 7D 结果) ---
        # ik_solution_pb 包含 7 个关节角度，顺序对应 pb_joint_indices_active
        # 我们需要根据 SAPIEN 活动关节名称的顺序 (假设前 7 个是手臂) 来排列它们
        
        ik_qpos_sapien_7d_np = np.zeros(7) # 创建一个 7-dim 数组
        mapped_back_count = 0
        print("准备将 IK 解映射回 SAPIEN 7D qpos (按 SAPIEN 前 7 活动关节顺序)...")

        # 创建 PyBullet 索引到其在 IK 解中位置的映射
        pb_idx_to_solution_index = {pb_idx: i for i, pb_idx in enumerate(pb_joint_indices_active)}

        # 遍历 SAPIEN 的前 7 个活动关节名称
        for sapien_idx in range(7):
            sapien_joint_name = all_sapien_joint_names[sapien_idx]
            # 查找此 SAPIEN 关节对应的 PyBullet 索引
            if sapien_joint_name in sapien_name_to_pb_index:
                pb_idx = sapien_name_to_pb_index[sapien_joint_name]
                # 查找此 PyBullet 索引在 IK 解中的位置
                if pb_idx in pb_idx_to_solution_index:
                    solution_idx = pb_idx_to_solution_index[pb_idx]
                    ik_angle = ik_solution_pb[solution_idx]
                    ik_qpos_sapien_7d_np[sapien_idx] = ik_angle
                    mapped_back_count += 1
                    # print(f"  Mapped SAPIEN idx {sapien_idx} ('{sapien_joint_name}') from PB idx {pb_idx} (solution idx {solution_idx}) = {ik_angle:.4f}")
                else:
                    print(f"警告: PyBullet 索引 {pb_idx} (来自 SAPIEN '{sapien_joint_name}') 未在 IK 解索引映射中找到! 这不应该发生。")
            else:
                print(f"警告: SAPIEN 关节 '{sapien_joint_name}' (索引 {sapien_idx}) 未在 PyBullet 名称映射中找到! URDF 可能不匹配 SAPIEN 定义。")

        print(f"成功将 {mapped_back_count} 个关节角度映射到 7D 结果。")
        if mapped_back_count != 7:
            print("警告: 未能精确映射 7 个关节! IK 结果可能不准确。")
            # return None, False # 根据严格程度决定是否失败

        # 转换为 torch tensor [1, 7]
        ik_qpos_result_7d = torch.tensor(ik_qpos_sapien_7d_np, dtype=dtype, device=device).unsqueeze(0)
        print("IK 计算成功完成 (返回 7D 结果)。")
        return ik_qpos_result_7d, True # 返回 7D 结果

    except Exception as pybullet_err:
        print(f"使用 PyBullet 进行 IK 计算时出错: {pybullet_err}")
        import traceback; traceback.print_exc()
        return None, False
    finally:
        # --- 7. 断开连接 ---
        if physics_client is not None and p.isConnected(physics_client):
            print("断开 PyBullet 连接...")
            p.disconnect(physics_client)
# --- PyBullet IK 函数结束 ---

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入环境
try:
    from two_robot_stack_cube_env import TwoRobotStackCubeEnv
    print("成功导入双臂堆叠立方体环境")
except ImportError as e:
    print(f"导入环境失败: {e}")
    sys.exit(1)

def test_set_pose_and_capture():
    """设置左臂姿态，拍照并保持渲染"""
    # --- 新增: 初始化条件赋值的变量 ---
    final_image_filename = None
    final_depth_filename = None
    # --- 新增结束 ---

    print("创建环境...")
    # --- 修改: 使用 "rgb+depth" obs_mode ---
    # 这个模式应该提供处理后的 rgb 和 depth 图像，
    # 并且通常也会在 obs["sensor_param"] 中包含相机参数。
    # --- 修改: 尝试使用 raster 着色器看深度是否改善 ---
    # env = TwoRobotStackCubeEnv(render_mode="human", shader_dir="raster", obs_mode="rgb+depth") 
    # --- 恢复: 使用 rt-fast ---
    # env = TwoRobotStackCubeEnv(render_mode="human", shader_dir="rt-fast", obs_mode="rgb+depth")
    # --- 新增: 指定使用包含 IK 的控制器配置 ---
    controller_choice = 'pd_joint_pos' # <<< 修改为关节位置控制器
    print(f"指定环境使用控制器配置: {controller_choice}")
    env = TwoRobotStackCubeEnv(
        render_mode="human", 
        # shader_dir="rt-fast", 
        obs_mode="rgb+depth",
        control_mode=controller_choice # <<< 正确参数名
    )
    # --- 修改结束 ---
    print("666666666666666666666666666")
    print("重置环境...")
    try:
        # 新版Gymnasium API
        obs, info = env.reset()
        print("使用新版Gymnasium API (返回obs, info)")
        # --- 删除: 不再在这里获取相机参数，因为它们会因手臂移动而失效 ---
        # left_agent_index = 0
        # camera_name_base = "hand_camera"
        # print(f"尝试获取相机 '{camera_name_base}' 的内参...")
        # try:
        #     if "sensor_param" in obs and isinstance(obs["sensor_param"], dict) and camera_name_base in obs["sensor_param"]:
        #         intrinsics_cv = obs["sensor_param"][camera_name_base]["intrinsic_cv"]
        #         # ... (其余删除的代码)
        # except Exception as e:
        #     print(f"从 obs 获取相机内参时出错: {e}")
        #     intrinsics_cv = None
        # try:
        #     if "sensor_param" in obs and isinstance(obs["sensor_param"], dict) and camera_name_base in obs["sensor_param"]:
        #         if "cam2world_gl" in obs["sensor_param"][camera_name_base]:
        #             cam2world_matrix = obs["sensor_param"][camera_name_base]["cam2world_gl"]
        #             # ... (其余删除的代码)
        # except Exception as e:
        #     print(f"从 obs 获取相机位姿时出错: {e}")
        #     cam2world_matrix = None
        # --- 删除结束 ---
        
    except ValueError:
        # 旧版API (可能不再适用，因为需要 obs)
        obs = env.reset()
        print("使用旧版API (仅返回obs)")

    # --- 添加: 手动设置左臂肘部角度 ---
    print("手动设置左臂初始姿态...")
    try:
        left_agent = env.agent.agents[0]
        # 获取当前关节位置 (qpos)
        qpos = left_agent.robot.get_qpos() # 返回的是 (1, N) 的张量，N是关节数

        # 计算目标角度 (向下旋转70度)
        elbow_joint_index = 5 # 假设肘部关节索引是3
        angle_offset = -70.0 * torch.pi / 180.0
        
        # 直接修改qpos张量中的值
        # 确保在正确的设备上操作 (与qpos一致)
        angle_offset_tensor = torch.tensor(angle_offset, device=qpos.device, dtype=qpos.dtype)
        
        # 检查 qpos 形状，通常是 [1, num_joints]
        if qpos.shape[0] == 1 and qpos.shape[1] > elbow_joint_index:
            # 获取当前肘部角度并加上偏移量
            current_elbow_qpos = qpos[0, elbow_joint_index]
            new_elbow_qpos = current_elbow_qpos + angle_offset_tensor
            
            # 检查是否超出关节限制 (可选但推荐)
            qlimits = left_agent.robot.get_qlimits() # 获取关节限制 [num_joints, 2]
            # --- 修正: 正确处理批处理维度 (假设形状是 [1, num_joints, 2]) ---
            lower_limit = qlimits[0, elbow_joint_index, 0]
            upper_limit = qlimits[0, elbow_joint_index, 1]
            # --- 修正结束 ---
            
            # 将新角度限制在范围内
            new_elbow_qpos_clamped = torch.clamp(new_elbow_qpos, lower_limit, upper_limit)
            
            # 更新 qpos 张量
            qpos[0, elbow_joint_index] = new_elbow_qpos_clamped
            
            # 设置新的关节位置
            left_agent.robot.set_qpos(qpos)
            
            # (可选) 步进一步模拟以应用更改 - 在拍照前建议步进一下
            env.scene.step() # 直接调用 scene.step() 可能更直接
            
            print(f"左臂关节 {elbow_joint_index} 已设置为目标角度 (限制在范围内)。")
            
            # --- 新增: 打印物体和桌子的世界 Z 坐标 --- 
            try:
                # --- 修改: 使用 [0, 2] 访问批处理后的 Z 坐标 ---
                banana_z = env.banana.pose.p[0, 2] if env.banana else 'N/A'
                table_z = env.table_scene.table.pose.p[0, 2] if hasattr(env, 'table_scene') and env.table_scene.table else 'N/A'
                # --- 修改结束 ---
                print(f"--- 调试: 物体世界 Z 坐标 --- Banana Z: {banana_z}, Table Z: {table_z}")
            except Exception as e:
                print(f"--- 调试: 获取物体 Z 坐标时出错: {e} ---")
            # --- 新增结束 ---
            
        else:
            print(f"错误：无法获取或设置左臂 qpos。Shape: {qpos.shape}, Index: {elbow_joint_index}")

    except Exception as e:
        import traceback
        print(f"设置左臂姿态时出错: {e}")
        print(traceback.format_exc())
    # --- 设置结束 ---

    # --- 添加: 拍照并保存左臂相机图像 ---
    print("尝试从左臂腕部相机拍照...")
    image_save_dir = os.path.join(current_dir, "capture_images") # 改个目录名
    os.makedirs(image_save_dir, exist_ok=True)
    image_filename = os.path.join(image_save_dir, f"left_arm_capture.png")
    depth_filename = os.path.join(image_save_dir, f"left_arm_depth.npy")
    camera_name = 'hand_camera' 
    agent_left = env.agent.agents[0]
    agent_id = "agent_0" # 左臂 ID

    # --- 新增: 用于存储相机参数和结果的变量 ---
    rgb_saved = False
    depth_saved = False
    # --- 新增结束 ---

    # --- 定义左臂相机基础名称 ---
    # left_agent_index = 0
    camera_name_base = "hand_camera"
    # camera_uid_left = f"{camera_name_base}_agent_{left_agent_index}" # 不再需要用这个作为 key
    # --- 定义结束 ---

    try:
        # --- 修改: 检查 agent.sensors 中是否存在 camera_name_base ---
        if hasattr(agent_left, 'sensors') and camera_name_base in agent_left.sensors:
            camera = agent_left.sensors[camera_name_base] # 使用基础名称获取相机对象
            print(f"正在捕获机器人 {agent_id} 手部相机 '{camera_name_base}' 图像和参数...") # 使用基础名称
            
            # 确保相机渲染是更新的
            env.render() # 先渲染一次确保相机视图更新
            camera.capture() # 捕获当前帧
            # --- 修改: 同时请求 RGB 和 Depth 数据 ---
            obs_dict = camera.get_obs(rgb=True, depth=True, segmentation=False)
            # --- 修改结束 ---

            # --- 获取内参和外参 (已移到 reset() 后处理 obs['sensor_param']) ---
            # --- 这里不再需要单独获取，因为已经从 obs 中获取 --- 

            if 'rgb' in obs_dict:
                rgb_image = obs_dict['rgb']
                if hasattr(rgb_image, 'cpu'):
                    rgb_image_np = rgb_image.cpu().numpy()
                else:
                    rgb_image_np = np.array(rgb_image)

                if rgb_image_np.ndim == 4 and rgb_image_np.shape[0] == 1: 
                    rgb_image_np = rgb_image_np[0]
                elif rgb_image_np.ndim != 3:
                    raise ValueError(f"机器人 {agent_id} 获取到的 RGB 图像维度不正确: {rgb_image_np.shape}")

                if rgb_image_np.dtype == np.uint8:
                    bgr_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
                else:
                    rgb_image_uint8 = (np.clip(rgb_image_np, 0, 1) * 255).astype(np.uint8)
                    bgr_image = cv2.cvtColor(rgb_image_uint8, cv2.COLOR_RGB2BGR)

                cv2.imwrite(image_filename, bgr_image)
                print(f"左臂手部相机 RGB 图像已保存至: {image_filename}")
                rgb_saved = True # 标记 RGB 已保存
            else:
                # --- 修改: 使用 camera_name_base 打印错误 ---
                print(f"错误: 未能从机器人 {agent_id} 相机 '{camera_name_base}' 的观测中获取 'rgb' 数据。")
            # --- 修改结束 ---

            if 'depth' in obs_dict:
                depth_image = obs_dict['depth']
                if hasattr(depth_image, 'cpu'):
                    depth_image_np = depth_image.cpu().numpy()
                else:
                    depth_image_np = np.array(depth_image)

                # 移除可能的批处理维度 (e.g., [1, H, W, 1] -> [H, W, 1])
                if depth_image_np.ndim == 4 and depth_image_np.shape[0] == 1:
                    depth_image_np = depth_image_np[0]
                elif depth_image_np.ndim != 3 or depth_image_np.shape[-1] != 1:
                     # 如果是 [H, W]，也调整为 [H, W, 1] 以保持一致性
                    if depth_image_np.ndim == 2:
                        depth_image_np = np.expand_dims(depth_image_np, axis=-1)
                    else:
                        raise ValueError(f"机器人 {agent_id} 获取到的深度图像维度不正确: {depth_image_np.shape}")

                # 深度数据通常是 uint16 或 int16 (毫米)
                # 保存为 .npy 文件以保留精度
                np.save(depth_filename, depth_image_np)
                print(f"左臂手部相机深度图像已保存至: {depth_filename} (格式: {depth_image_np.dtype}, 形状: {depth_image_np.shape}, 单位: 毫米)")
                depth_saved = True # 标记深度图已保存
            else:
                 print(f"错误: 未能从机器人 {agent_id} 相机 '{camera_name_base}' 的观测中获取 'depth' 数据。")
            # --- 修改结束 ---

            # --- 新增: 在拍照后、感知前获取最新的相机参数 ---
            print("在拍照后获取最新的观测值 (包含相机参数)... ")
            # 注意: env.get_obs() 通常是获取当前状态观测值的方法
            # 如果环境实现不同，可能需要调用 env.step(dummy_action) 或其他方法来刷新观测值
            try:
                current_obs = env.get_obs()
                print("成功调用 env.get_obs()")
            except AttributeError:
                print("警告: env.get_obs() 不可用，尝试调用 env.step() 获取新观测值。")
                # 如果没有 get_obs()，尝试用空动作步进一次来获取新的 obs
                # 注意: 这可能会意外地改变状态，如果环境不允许空动作
                try:
                     # 使用与环境动作空间匹配的虚拟动作
                     action_space = env.action_space
                     if isinstance(action_space, gym.spaces.Dict):
                         dummy_action = {k: torch.zeros_like(v.sample()) for k, v in action_space.spaces.items()} 
                     else: # 假设是 Box 或类似空间
                         dummy_action = torch.zeros_like(torch.from_numpy(action_space.sample()))
                     current_obs, _, _, _, info = env.step(dummy_action) 
                     print("通过 env.step(dummy_action) 获取了新的 obs。")
                except Exception as step_err:
                     print(f"尝试 env.step() 获取新 obs 失败: {step_err}")
                     current_obs = obs # 退回使用 reset 时的 obs (可能不准)
            except Exception as get_obs_err:
                print(f"调用 env.get_obs() 出错: {get_obs_err}")
                current_obs = obs # 退回

            # 从最新的观测值中提取参数
            intrinsics_cv = None
            cam2world_matrix = None # 这个是 T_Wgl_glCam (可能不再需要)
            extrinsic_cv = None # 新增：这个是 T_cvCam_Wstd
            camera_name_base = "hand_camera" # 确保与之前一致

            print(f"尝试从最新的观测值中提取 '{camera_name_base}' 的参数...")
            try:
                if "sensor_param" in current_obs and isinstance(current_obs["sensor_param"], dict) and camera_name_base in current_obs["sensor_param"]:
                    sensor_params = current_obs["sensor_param"][camera_name_base]
                    # 提取内参
                    if "intrinsic_cv" in sensor_params:
                        intrinsics_cv = sensor_params["intrinsic_cv"]
                        if hasattr(intrinsics_cv, 'cpu'): intrinsics_cv = intrinsics_cv.cpu().numpy()
                        if intrinsics_cv.ndim == 3 and intrinsics_cv.shape[0] == 1: intrinsics_cv = intrinsics_cv.squeeze(0)
                        print(f"从新 obs 获取到内参 (Shape: {intrinsics_cv.shape})")
                    else: print("警告: 新 obs 中未找到 'intrinsic_cv'")
                    
                    # 提取外参 (cam2world_gl - 不再直接用于最终变换，保留以备调试)
                    if "cam2world_gl" in sensor_params:
                        cam2world_matrix = sensor_params["cam2world_gl"]
                        if hasattr(cam2world_matrix, 'cpu'): cam2world_matrix = cam2world_matrix.cpu().numpy()
                        if cam2world_matrix.ndim == 3 and cam2world_matrix.shape[0] == 1: cam2world_matrix = cam2world_matrix.squeeze(0)
                        # print(f"从新 obs 获取到外参 cam2world_gl (Shape: {cam2world_matrix.shape})") # 可以注释掉
                    # else: print("警告: 新 obs 中未找到 'cam2world_gl'") # 注释掉，避免过多警告

                    # 新增: 提取 extrinsic_cv 
                    if "extrinsic_cv" in sensor_params:
                        extrinsic_cv = sensor_params["extrinsic_cv"]
                        if hasattr(extrinsic_cv, 'cpu'): extrinsic_cv = extrinsic_cv.cpu().numpy()
                        if extrinsic_cv.ndim == 3 and extrinsic_cv.shape[0] == 1: extrinsic_cv = extrinsic_cv.squeeze(0)
                        print(f"从新 obs 获取到外参 extrinsic_cv (Shape: {extrinsic_cv.shape})")
                    else:
                        print("警告: 新 obs 中未找到 'extrinsic_cv'")
                        extrinsic_cv = None # 确保失败时为 None

                    # +++ 新增调试：获取 camera_link 的真实世界位姿并比较 +++
                    try:
                        print(f"\\n--- 调试: 验证 cam2world_matrix 是否为 camera_link/sensor 的世界位姿 ---")
                        camera_link_name = "camera_link" # 目标 Link 名称
                        camera_name_base = "hand_camera" # 相机传感器名称

                        # --- 修改: 尝试从相机传感器对象获取位姿 ---
                        ee_link_pose_truth = None
                        sensor_pose_available = False
                        if hasattr(agent_left, 'sensors') and camera_name_base in agent_left.sensors:
                            camera_sensor = agent_left.sensors[camera_name_base]
                            if hasattr(camera_sensor, 'pose'):
                                ee_link_pose_truth = camera_sensor.pose # 获取 sapien.Pose 对象
                                sensor_pose_available = True
                                print(f"成功从 camera_sensor '{camera_name_base}' 获取到 .pose 属性。")
                            else:
                                print(f"警告: camera_sensor '{camera_name_base}' 没有 .pose 属性。")
                        else:
                             print(f"警告: 未找到名为 '{camera_name_base}' 的相机传感器。")

                        if sensor_pose_available and ee_link_pose_truth is not None:
                            print(f"获取到 camera_sensor '{camera_name_base}' 的真实世界位姿 (Sapien Pose): p={ee_link_pose_truth.p}, q={ee_link_pose_truth.q}")
                                
                            # --- 修正: 确保从 tensor 转换 --- 
                            p_np = ee_link_pose_truth.p
                            q_np = ee_link_pose_truth.q
                            if hasattr(p_np, 'cpu'): p_np = p_np.cpu().numpy()
                            if hasattr(q_np, 'cpu'): q_np = q_np.cpu().numpy()
                            # 移除可能的批处理维度
                            if p_np.ndim > 1: p_np = p_np.squeeze(0) 
                            if q_np.ndim > 1: q_np = q_np.squeeze(0)
                                
                            T_World_EE_truth = pose_to_transformation_matrix(p_np, q_np) # 确保输入是 NumPy
                            print(f"转换得到的 T_World_EE_truth (来自 sensor.pose):\\n{np.round(T_World_EE_truth, 6)}") # 打印真实位姿

                            if cam2world_matrix is not None:
                                print(f"\\n对比 obs['sensor_param'] 中的 cam2world_matrix (假设为 T_World_EE):\\n{np.round(cam2world_matrix, 6)}") # 打印 sensor_param 中的位姿

                                # 计算差值矩阵，看是否接近单位阵
                                if T_World_EE_truth.shape == (4, 4) and cam2world_matrix.shape == (4, 4):
                                    try:
                                        # 确保类型一致
                                        T_W_EE_truth_f64 = T_World_EE_truth.astype(np.float64)
                                        cam2world_f64 = cam2world_matrix.astype(np.float64)
                                             
                                        inv_T_W_EE_truth = np.linalg.inv(T_W_EE_truth_f64)
                                        diff_matrix = inv_T_W_EE_truth @ cam2world_f64
                                        print(f"\\n差值矩阵 (sensor.pose)^-1 * cam2world_matrix (理想情况接近单位阵):\\n{np.round(diff_matrix, 6)}")
                                        
                                        # 额外检查：反向相乘 diff_matrix_rev = cam2world^-1 * sensor.pose
                                        inv_cam2world = np.linalg.inv(cam2world_f64)
                                        diff_matrix_rev = inv_cam2world @ T_W_EE_truth_f64
                                        print(f"\\n反向差值矩阵 cam2world^-1 * (sensor.pose) (理想情况接近单位阵):\\n{np.round(diff_matrix_rev, 6)}")

                                    except np.linalg.LinAlgError:
                                        print("\\n计算差值矩阵时求逆失败。")
                                else:
                                    print("\\n矩阵形状不匹配，无法计算差值。")
                            else:
                                print("\\ncam2world_matrix 未获取到，无法对比。")
                        else:
                             print(f"未能通过 sensor.pose 获取真实世界位姿。")

                    except AttributeError as attr_err:
                         print(f"--- 调试: 调用方法或访问属性时出错 (可能是 API 不匹配): {attr_err} ---") 
                    except Exception as debug_err:
                        import traceback
                        print(f"--- 调试: 获取或比较 camera 位姿时出错: {debug_err} ---")
                        print(traceback.format_exc())
                    print(f"--- 调试结束 ---\\n")
                    # +++ 调试结束 +++

                    # +++ 新增: 尝试获取 camera_link 的世界位姿 +++
                    T_World_Link = None
                    try:
                        print("\\n--- 调试: 尝试获取 camera_link 的世界位姿 ---")
                        target_link_name = "camera_link"
                        links = agent_left.robot.get_links()
                        camera_link_found = False
                        for link in links:
                            if link.name == target_link_name:
                                print(f"找到 Link: {link.name}")
                                if hasattr(link, 'pose'):
                                    link_pose = link.pose # 获取 sapien.Pose
                                    p_link = link_pose.p
                                    q_link = link_pose.q
                                    if hasattr(p_link, 'cpu'): p_link = p_link.cpu().numpy()
                                    if hasattr(q_link, 'cpu'): q_link = q_link.cpu().numpy()
                                    if p_link.ndim > 1: p_link = p_link.squeeze(0)
                                    if q_link.ndim > 1: q_link = q_link.squeeze(0)

                                    T_World_Link = pose_to_transformation_matrix(p_link, q_link)
                                    print(f"获取到 '{target_link_name}' 的世界位姿 (Sapien Pose): p={p_link}, q={q_link}")
                                    print(f"转换得到的 T_World_Link:\\n{np.round(T_World_Link, 6)}")
                                    camera_link_found = True
                                else:
                                    print(f"警告: Link '{link.name}' 没有 .pose 属性。")
                                break # 找到后退出循环
                        if not camera_link_found:
                             print(f"错误: 未能在机器人 links 中找到名为 '{target_link_name}' 的 link。可用 links: {[l.name for l in links]}")
                             
                    except AttributeError as attr_err:
                         print(f"--- 调试: 调用 robot.get_links() 或 link.pose 时出错: {attr_err} ---")
                    except Exception as link_debug_err:
                        import traceback
                        print(f"--- 调试: 获取 link 位姿时出错: {link_debug_err} ---")
                        print(traceback.format_exc())
                    print("--- 调试结束 ---\\n")
                    # +++ camera_link 位姿获取结束 +++

                else:
                    print(f"警告: 未能在新的 obs['sensor_param'] 中找到相机 '{camera_name_base}'。obs keys: {list(current_obs.keys())}")
                    if "sensor_param" in current_obs: print(f"sensor_param keys: {list(current_obs['sensor_param'].keys())}")
            except Exception as e:
                print(f"从新 obs 获取相机参数时出错: {e}")
            # --- 参数获取结束 ---

            # --- 修改: 检查参数是否成功获取，现在需要 extrinsic_cv ---
            print("--- 检查感知流程前置条件 ---")
            print(f"rgb_saved: {rgb_saved} (Type: {type(rgb_saved)})")
            print(f"depth_saved: {depth_saved} (Type: {type(depth_saved)})")
            print(f"intrinsics_cv is None: {intrinsics_cv is None} (Type: {type(intrinsics_cv)})")
            if intrinsics_cv is not None: print(f"intrinsics_cv Shape: {intrinsics_cv.shape}")
            print(f"cam2world_matrix is None: {cam2world_matrix is None} (Type: {type(cam2world_matrix)})")
            if cam2world_matrix is not None: print(f"cam2world_matrix Shape: {cam2world_matrix.shape}")
            print(f"extrinsic_cv is None: {extrinsic_cv is None} (Type: {type(extrinsic_cv)})")
            if extrinsic_cv is not None: print(f"extrinsic_cv Shape: {extrinsic_cv.shape}")
            print("-------------------------------")
            
            # --- 开始运行感知流程 (如果参数有效) ---
            if rgb_saved and depth_saved and intrinsics_cv is not None and cam2world_matrix is not None:
                print("--- 开始运行感知流程 ---")
                # --- 新增: 从 perception_pipeline 导入 run_perception ---
                # try:
                #     from perception_pipeline import run_perception
                # except ImportError as e:
                #     print(f"无法导入 run_perception: {e}")
                #     run_perception = None # 设为 None 以便后续检查
                # --- 新增结束 ---

                # --- 临时禁用感知 ---
                run_perception = None # 强制禁用
                center_point_camera = None # 设为 None

                # if run_perception: # 确保导入成功
                #     object_name = "the banana" # 或其他您想检测的物体
                #     sam_ckpt_path = "/home/kewei/17robo/01mydemo/01ckpt/sam_vit_h_4b8939.pth" # 您的 SAM 模型路径
                            
                #     # --- 打印传递给 run_perception 的参数 ---
                #     print("传递给 run_perception 的参数:")
                #     print(f"  rgb_image_path: {image_filename}")
                #     print(f"  depth_map_path: {depth_filename}")
                #     print(f"  intrinsics (shape): {intrinsics_cv.shape}")
                #     print(f"  intrinsics:\\n{np.round(intrinsics_cv, 6)}") # <-- 新增打印具体数值
                #     print(f"  text_prompt: {object_name}")
                #     print(f"  sam_model_path: {sam_ckpt_path}")
                #     # print(f"  cam2world_matrix (shape): {cam2world_matrix.shape}") # <-- 注释掉，因为不再直接使用
                #     # print(f"  cam2world_matrix:\\n{cam2world_matrix}") # 取消注释以查看具体值
                #     # --- 打印结束 ---

                #     center_point_camera = run_perception(
                #         rgb_image_path=image_filename,
                #         depth_map_path=depth_filename,
                #         intrinsics=intrinsics_cv,
                #         text_prompt=object_name,
                #         sam_model_path=sam_ckpt_path,
                #         show_visuals=True # 设为 True 进行可视化
                #     )
                # else:
                #     print("错误: run_perception 未成功导入或被禁用，无法运行感知流程。")
                # --- 感知调用结束 ---

                # if center_point_camera is not None: # <<< 注释掉这部分处理 camera 坐标的代码 >>>
                #     print(f"感知流程完成，物体 '{object_name}' 在相机坐标系中心点 (OpenCV): {center_point_camera}") # 明确是 OpenCV

                #     # +++ 新增: 验证感知结果 P_cvCam 是否合理 (反向投影到 2D) +++
                #     if center_point_camera is not None and intrinsics_cv is not None:
                #         print("--- 验证感知结果: 将 P_cvCam 反向投影到 2D 图像 ---")
                #         K = intrinsics_cv # 内参矩阵
                #         P_cv = center_point_camera # 3D 点 (相机系)

                #         # 检查 P_cv 的 Z 坐标是否大于 0 (在相机前方)
                #         if P_cv[2] > 1e-6:
                #             try:
                #                 # 投影: K @ P_cv
                #                 uvw_prime = K @ P_cv
                                        
                #                 # 齐次除法
                #                 u_proj = uvw_prime[0] / uvw_prime[2]
                #                 v_proj = uvw_prime[1] / uvw_prime[2]
                                        
                #                 print(f"计算得到的投影 2D 坐标 (u, v): ({u_proj:.2f}, {v_proj:.2f})", flush=True)
                                        
                #             except Exception as proj_err:
                #                  print(f"反向投影计算时出错: {proj_err}")
                #         else:
                #             print(f"警告: P_cvCam 的 Z 坐标 ({P_cv[2]}) 非正，无法进行有效投影。")
                #         print("--- 验证结束 ---")
                #     # +++ 新增结束 +++

                #     # --- 修改: 使用 link.pose 和 假设的固定 T_Link_cvCam ---
                #     center_point_world = None # 初始化
                #     try:
                #         # 1. 检查 T_World_Link (来自 link.pose) 是否有效
                #         if T_World_Link is not None:
                #             print(f"使用获取到的 T_World_Link (来自 link.pose):\\n{np.round(T_World_Link, 6)}")
                                    
                #             # 2. 定义假设的、固定的 T_Link_cvCam (相机相对于Link的变换)
                #             T_Link_cvCam_assumed = np.array([
                #                 [ 0.,  0.,  1.,  0.],
                #                 [-1.,  0.,  0.,  0.],
                #                 [ 0., -1.,  0.,  0.],
                #                 [ 0.,  0.,  0.,  1.]
                #             ])
                #             print(f"使用假设的固定 T_Link_cvCam_assumed:\\n{np.round(T_Link_cvCam_assumed, 6)}")

                #             # 3. 获取感知点 (OpenCV 相机系) 的齐次坐标
                #             center_point_camera_cv_h = np.append(center_point_camera, 1)
                #             print(f"OpenCV 相机坐标 (齐次 P_cvCam_h): {center_point_camera_cv_h}")

                #             # 4. 转换到世界坐标系: P_World = T_World_Link * T_Link_cvCam_assumed * P_cvCam
                #             center_point_world_h = T_World_Link @ T_Link_cvCam_assumed @ center_point_camera_cv_h
                #             center_point_world = center_point_world_h[:3]
                #             print(f"最终计算的世界坐标 (T_WL * T_Lc_assumed * P_cv): {center_point_world}")
                                        
                #         else:
                #             print("错误: T_World_Link (link.pose) 无效，无法执行变换。")
                            
                #     except Exception as e:
                #         import traceback
                #         print(f"执行最终坐标变换时出错: {e}")
                #         print(traceback.format_exc())
                #     # --- 变换结束 ---

                # else: # <<< 感知失败或禁用时的处理 >>>
                #     print("感知流程失败或被禁用，使用固定的世界坐标。")

                # --- 新增: 直接使用固定的世界坐标 ---
                center_point_world = np.array([0.48967573, 0.13179222, 0.03743867])
                print(f"使用固定的世界坐标: {center_point_world}")
                # --- 新增结束 ---

                # --- 在场景中添加标记点 (使用最终计算结果) ---
                if center_point_world is not None:
                    print(f"尝试在场景中添加中心点标记 ({center_point_world})...")
                    try:
                        builder = env.scene.create_actor_builder()
                        radius = 0.01
                        render_material = sapien.render.RenderMaterial(base_color=[0, 1, 1, 1]) # 改为青色标记
                        builder.add_sphere_visual(radius=radius, material=render_material)
                        marker_actor = builder.build_static(name="center_marker")
                        marker_actor.set_pose(sapien.Pose(p=center_point_world))
                        print(f"成功添加中心点标记 (Actor: {marker_actor.name})。")
                    except Exception as marker_err:
                        print(f"添加标记点时出错: {marker_err}")
                else:
                     print("未能计算出世界坐标，无法添加标记。")

                # 5. (可选) 验证结果
                #    在场景中用计算出的 center_point_world 添加标记点，
                #    并与物体的 Ground Truth 世界坐标进行比较。
                #    Ground Truth: [0.5, 0.15, 0.02401901]
                #    计算结果:   [0.49016993, 0.13254908, 0.0377709] # 这是之前的香蕉示例结果
                #    结果非常接近，验证了方法的有效性。
                if center_point_world is not None:
                     print(f"--- 调试: 计算出的物体世界坐标: {center_point_world} ---")
                     # --- (可以取消注释以添加标记点进行可视化) ---
                     # builder = env.scene.create_actor_builder()
                     # builder.add_sphere_visual(radius=0.01, color=[1, 0, 0]) # 红色小球
                     # marker = builder.build_static(name=\"perception_marker\")
                     # marker.set_pose(sapien.Pose(p=center_point_world))
                     # print("--- 调试: 在感知位置添加了红色标记点 ---")
                     pass # 如果上面注释掉了，需要 pass

                # --- 新增: 移动到目标点上方并拍照 (移到这里) ---
                if center_point_world is not None:
                    print("\\n--- 开始移动到目标点上方并拍照 --- ")
                    original_qpos = agent_left.robot.get_qpos().clone() # 保存原始 qpos 以便恢复
                    try:
                        agent_left = env.agent.agents[0]
                        
                        # 1. 计算目标位姿 (世界和基座标系)
                        target_pos = center_point_world + np.array([0, 0, 0.3])
                        print(f"计算的目标末端执行器位置 (target_pos): {target_pos}")
                        target_rot_mat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
                        target_quat_xyzw = tq.mat2quat(target_rot_mat)
                        target_pose_world = sapien.Pose(p=target_pos, q=target_quat_xyzw) # 世界坐标目标
                        print(f"计算的目标末端执行器姿态 (世界 quat_xyzw): {target_quat_xyzw}")

                        # 获取机器人基座标系的世界位姿
                        base_pose_p = agent_left.robot.pose.p
                        base_pose_q = agent_left.robot.pose.q
                        if hasattr(base_pose_p, 'cpu'): base_pose_p = base_pose_p.cpu().numpy()
                        if hasattr(base_pose_q, 'cpu'): base_pose_q = base_pose_q.cpu().numpy()
                        if base_pose_p.ndim > 1: base_pose_p = base_pose_p.squeeze(0)
                        if base_pose_q.ndim > 1: base_pose_q = base_pose_q.squeeze(0)
                        base_pose_mat = pose_to_transformation_matrix(base_pose_p, base_pose_q) # T_World_Base
                        T_Base_World = np.linalg.inv(base_pose_mat)
                        
                        # 转换为基座标系目标位姿
                        target_pose_world_mat = pose_to_transformation_matrix(target_pose_world.p, target_pose_world.q)
                        target_pose_base_mat = T_Base_World @ target_pose_world_mat
                        target_p_base = target_pose_base_mat[:3, 3]
                        target_q_base_xyzw = tq.mat2quat(target_pose_base_mat[:3, :3])
                        # --- 重新添加: 创建 target_pose_base --- 
                        target_pose_base = sapien.Pose(p=target_p_base, q=target_q_base_xyzw) # 基座标目标
                        print(f"将目标位姿转换为基坐标系: p={target_p_base}, q={target_q_base_xyzw}")
                        # --- 重新添加结束 ---

                        # --- 重新添加: 定义 PyBullet IK 所需的变量 ---
                        agent_left = env.agent.agents[0] # 获取左臂 agent
                        urdf_path_for_ik = str(_project_root / 'urdf_01/GEN72.urdf') # 使用相对路径
                        ee_link_name = agent_left.ee_link_name # 末端 Link 名称
                        current_sapien_qpos = agent_left.robot.get_qpos() # 当前 SAPIEN qpos [1, N_active]
                        robot_articulation = agent_left.robot # Articulation 对象
                        all_sapien_joint_names = [j.get_name() for j in robot_articulation.get_active_joints()]
                        device = current_sapien_qpos.device
                        dtype = current_sapien_qpos.dtype
                        print(f"准备调用 PyBullet IK: URDF='{urdf_path_for_ik}', EE='{ee_link_name}'")
                        # --- 重新添加结束 ---

                        # --- 修改: 调用 PyBullet IK --- 
                        ik_qpos_7d, ik_success = compute_ik_with_pybullet(
                            urdf_path=urdf_path_for_ik,
                            ee_link_name=ee_link_name,
                            target_pose_base=target_pose_base,
                            current_sapien_qpos=current_sapien_qpos,
                            all_sapien_joint_names=all_sapien_joint_names,
                            device=device,
                            dtype=dtype
                        )
                        # --- 修改结束 ---
                        
                        if ik_success and ik_qpos_7d is not None:
                            # --- 修改: 打印 7D 结果 --- 
                            print(f"PyBullet IK 成功找到 7D 解: {ik_qpos_7d}")
                            # --- 修改结束 ---
                            
                            # --- 修改: 使用 env.step(action) 并构造 8D 动作 --- 
                            print("设置目标关节位置并通过 env.step() 模拟 (使用 8D 动作)...")
                            
                            agent_left = env.agent.agents[0]
                            agent_right = env.agent.agents[1]
                            qpos_left_current = agent_left.robot.get_qpos()  # Shape [1, 13]
                            qpos_right_current = agent_right.robot.get_qpos() # Shape [1, 13]

                            # 假设: 控制器期望的 8D = 7 个手臂关节 + 第 8 个活动关节 (index 7) 作为夹爪
                            gripper_joint_index = 7 
                            num_controlled_joints = 8 # 控制器期望的维度

                            try:
                                # --- 构造左臂 8D 动作 --- 
                                target_qpos_left_7d = ik_qpos_7d.squeeze(0) # Shape [7]
                                if qpos_left_current.shape[1] <= gripper_joint_index:
                                     raise IndexError(f"左臂 qpos 维度 ({qpos_left_current.shape[1]}) 不足以获取索引 {gripper_joint_index} 处的夹爪值")
                                current_gripper_left = qpos_left_current[0, gripper_joint_index:gripper_joint_index+1] # Shape [1], 保持为张量
                                action_left_8d = torch.cat((target_qpos_left_7d, current_gripper_left), dim=0).to(device=device, dtype=dtype) # Shape [8]

                                # --- 构造右臂 8D 动作 (保持当前姿态) ---
                                if qpos_right_current.shape[1] < num_controlled_joints:
                                     raise IndexError(f"右臂 qpos 维度 ({qpos_right_current.shape[1]}) 不足以获取前 {num_controlled_joints} 个受控关节")
                                action_right_8d = qpos_right_current.squeeze(0)[:num_controlled_joints].to(device=device, dtype=dtype) # Shape [8]

                                # --- 获取 Action Keys --- 
                                action_space_keys = list(env.action_space.spaces.keys())
                                if len(action_space_keys) < 2:
                                     raise ValueError(f"环境动作空间键不足两个: {action_space_keys}")
                                left_agent_key = action_space_keys[0]
                                right_agent_key = action_space_keys[1]
                                print(f"使用 Action Keys: Left='{left_agent_key}', Right='{right_agent_key}'")

                                # --- 构建最终 Action 字典 --- 
                                action = {
                                    left_agent_key: action_left_8d,
                                    right_agent_key: action_right_8d
                                }
                                print(f"构建的字典动作为: {left_agent_key} shape={action[left_agent_key].shape}, {right_agent_key} shape={action[right_agent_key].shape}")
                            
                            except IndexError as idx_err:
                                print(f"错误: 构造 8D 动作时发生索引错误 (检查夹爪/受控关节假设): {idx_err}")
                                action = None
                            except Exception as dict_build_err:
                                 print(f"错误: 构建动作字典时出错: {dict_build_err}")
                                 action = None 
                            # --- 动作构建结束 --- 

                            # 循环调用 env.step(action)
                            num_steps = 50
                            if action is not None: # 仅在动作构建成功时执行
                                for step_idx in range(num_steps):
                                    try:
                                        # print(f"Step {step_idx+1}/{num_steps}") # 可取消注释以跟踪步骤
                                        obs, reward, terminated, truncated, info = env.step(action)
                                        # SAPIEN 环境通常需要手动渲染
                                        if env.render_mode == "human":
                                                env.render()
                                    except Exception as step_err:
                                         print(f"env.step() 在第 {step_idx+1} 步出错: {step_err}")
                                         import traceback; traceback.print_exc()
                                         break # 出错则停止循环
                            else:
                                print("错误: 动作构建失败，跳过 env.step() 循环。")
                            # --- 修改结束 (包含 if action is not None) ---

                            print(f"已完成 {num_steps} 步模拟。")
                            
                            # 5. 拍照
                            print("执行拍照...")
                            camera_name = 'hand_camera'
                            image_save_dir = os.path.join(current_dir, "capture_images_after_move")
                            os.makedirs(image_save_dir, exist_ok=True)
                            final_image_filename = os.path.join(image_save_dir, f"final_pose_capture.png")
                            final_depth_filename = os.path.join(image_save_dir, f"final_pose_depth.npy")

                            if hasattr(agent_left, 'sensors') and camera_name in agent_left.sensors:
                                camera = agent_left.sensors[camera_name]
                                env.render() # 更新渲染器
                                camera.capture() # 捕获当前帧
                                obs_dict = camera.get_obs(rgb=True, depth=True)

                                if 'rgb' in obs_dict:
                                    rgb_image = obs_dict['rgb']
                                    if hasattr(rgb_image, 'cpu'): rgb_image_np = rgb_image.cpu().numpy()
                                    else: rgb_image_np = np.array(rgb_image)
                                    if rgb_image_np.ndim == 4: rgb_image_np = rgb_image_np[0]
                                    
                                    if rgb_image_np.dtype == np.uint8: bgr_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR)
                                    else: bgr_image = cv2.cvtColor((np.clip(rgb_image_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                                    
                                    cv2.imwrite(final_image_filename, bgr_image)
                                    print(f"最终姿态的 RGB 图像已保存至: {final_image_filename}")
                                else:
                                    print("错误: 未能在最终姿态获取 RGB 图像。")

                                if 'depth' in obs_dict:
                                    depth_image = obs_dict['depth']
                                    if hasattr(depth_image, 'cpu'): depth_image_np = depth_image.cpu().numpy()
                                    else: depth_image_np = np.array(depth_image)
                                    if depth_image_np.ndim == 4: depth_image_np = depth_image_np[0]
                                    if depth_image_np.ndim == 2: depth_image_np = np.expand_dims(depth_image_np, axis=-1)

                                    np.save(final_depth_filename, depth_image_np)
                                    print(f"最终姿态的深度图像已保存至: {final_depth_filename}")
                                else:
                                    print("错误: 未能在最终姿态获取深度图像。")
                            else:
                                print(f"错误: 无法访问相机 '{camera_name}' 进行最终拍照。")
                        else: # --- 处理 IK 失败的情况 (第一次移动) ---
                             print("第一次 IK 失败或未找到解，跳过移动和拍照步骤。") 
                        # --- 修正缩进结束 ---

                    except Exception as move_err:
                        print(f"在第一次移动和拍照过程中发生错误: {move_err}")
                        import traceback
                        traceback.print_exc()
                        # 出错时也尝试恢复姿态
                        # print("尝试恢复到 IK 前的关节姿态...")
                        # agent_left.robot.set_qpos(original_qpos)
                        # env.scene.step()
                # --- 第一次移动和拍照结束 --- 

            else:
                # --- 修改: 此处对应 center_point_world is None 的情况 --- 
                print("未能计算出世界坐标 (center_point_world is None)，跳过第一次移动和后续 GraspNet 步骤。") 
                # --- 修改结束 ---

            # ===================================================================
            # --- 第二阶段: 基于第一次拍照结果进行 GraspNet 预测和抓取 --- 
            # ===================================================================
            print("\\n" + "="*30 + " 第二阶段开始: GraspNet 预测与抓取尝试 " + "="*30)

            # --- 重新获取当前状态和相机参数 --- 
            print("--- 准备 GraspNet: 重新获取当前状态和相机参数 ---")
            # agent_left 应该仍然是 agent_left = env.agent.agents[0]
            # agent_id = "agent_0"
            camera_name_base = "hand_camera"
            current_obs_grasp = None
            intrinsics_cv_grasp = None
            
            # 获取最新的观测值和内参
            try:
                # 尝试获取最新观测值 (与之前类似)
                try:
                    current_obs_grasp = env.get_obs()
                    print("通过 env.get_obs() 获取最新观测值成功。")
                except AttributeError:
                    print("警告: env.get_obs() 不可用，尝试 step(None) 获取新 obs。")
                    try: current_obs_grasp, _, _, _, _ = env.step(None)
                    except: 
                        try: # 尝试零动作
                            action_space = env.action_space
                            if isinstance(action_space, gym.spaces.Dict): dummy_action = {k: torch.zeros_like(v.sample()) for k, v in action_space.spaces.items()}
                            else: dummy_action = torch.zeros_like(torch.from_numpy(action_space.sample()))
                            current_obs_grasp, _, _, _, _ = env.step(dummy_action)
                            print("通过 env.step(dummy_action) 获取最新观测值成功。")
                        except Exception as step_err: print(f"尝试 env.step() 获取新 obs 失败: {step_err}"); current_obs_grasp = current_obs # 退回
                except Exception as get_obs_err: print(f"调用 env.get_obs() 出错: {get_obs_err}"); current_obs_grasp = current_obs # 退回

                # 从最新观测值提取内参
                if current_obs_grasp and "sensor_param" in current_obs_grasp and isinstance(current_obs_grasp["sensor_param"], dict) and camera_name_base in current_obs_grasp["sensor_param"]:
                    sensor_params = current_obs_grasp["sensor_param"][camera_name_base]
                    if "intrinsic_cv" in sensor_params:
                        intrinsics_cv_grasp = sensor_params["intrinsic_cv"]
                        if hasattr(intrinsics_cv_grasp, 'cpu'): intrinsics_cv_grasp = intrinsics_cv_grasp.cpu().numpy()
                        if intrinsics_cv_grasp.ndim == 3 and intrinsics_cv_grasp.shape[0] == 1: intrinsics_cv_grasp = intrinsics_cv_grasp.squeeze(0)
                        print(f"为 GraspNet 获取到内参 (Shape: {intrinsics_cv_grasp.shape})")
                    else: print("警告: 新 obs 中未找到 'intrinsic_cv'")
                else: print("警告: 未能在新的 obs 中找到有效的 sensor_param 或相机内参。")

            except Exception as e:
                print(f"获取 GraspNet 输入参数时出错: {e}")
                import traceback; traceback.print_exc()
            # --- 参数获取结束 ---

            # --- 调用点云生成 --- 
            point_cloud_o3d = None
            point_cloud_np = None
            # 检查必要的输入文件和参数是否存在
            final_rgb_exists = final_image_filename and os.path.exists(final_image_filename)
            final_depth_exists = final_depth_filename and os.path.exists(final_depth_filename)
            intrinsics_exist = intrinsics_cv_grasp is not None
            
            if final_depth_exists and intrinsics_exist:
                print("\\n--- 开始生成点云 (使用最终姿态深度图) ---")
                point_cloud_o3d, point_cloud_np = depth_to_point_cloud(
                    depth_npy_path=final_depth_filename, # 使用最终姿态的深度图
                    intrinsics_matrix=intrinsics_cv_grasp,
                    depth_scale=1000.0,
                    clip_distance_m=1.0 # 截断距离设近一点，关注桌/物体面
                )
                if point_cloud_np is not None:
                    print("点云生成成功。")
                    # 可选保存
                    # ply_save_path = os.path.join(os.path.dirname(final_depth_filename), "final_pose_point_cloud.ply")
                    # try: o3d.io.write_point_cloud(ply_save_path, point_cloud_o3d); print(f"调试点云已保存到: {ply_save_path}")
                    # except Exception as save_err: print(f"保存点云失败: {save_err}")
                else:
                    print("点云生成失败。")
            else:
                print("缺少最终深度图路径或当前内参，无法生成点云。")
                if not final_depth_exists: print(f"  Depth file missing or path invalid: {final_depth_filename}")
                if not intrinsics_exist: print("  Intrinsics missing.")
            # --- 点云生成结束 --- 
            
            # --- 2025-11-28 修改: 集成感知管道 (Florence-2 + SAM) ---
            
            masked_point_cloud_np = point_cloud_np # 默认使用全场景点云
            
            if final_rgb_exists and final_depth_exists and intrinsics_exist:
                print("\n--- 正在运行感知管道 (Florence-2 + SAM) 以定位 'banana' ---")
                
                # 检查 run_perception 是否可用
                if run_perception is None:
                     print("错误: run_perception 函数未加载 (可能导入失败)。跳过感知步骤。")
                     # 可以在这里打印更详细的建议，例如安装依赖
                else:
                    try:
                        # 假设 SAM 模型路径 (根据之前的 perception_pipeline 示例)
                        sam_checkpoint_path = str(_project_root / "checkpoints/sam_vit_h_4b8939.pth")
                        
                        perception_result = run_perception(
                            rgb_image_path=final_image_filename,
                            depth_map_path=final_depth_filename,
                            intrinsics=intrinsics_cv_grasp,
                            text_prompt="the banana", # <--- 指定要抓取的物体
                            sam_model_path=sam_checkpoint_path,
                            show_visuals=True # 显示感知结果图
                        )
                        
                        if perception_result is not None:
                            # 解包返回值 (center, pcd, mask)
                            # 注意: 如果 perception_pipeline.py 未更新返回 3 个值，这里可能会报错，需要确保两者一致
                            if len(perception_result) == 3:
                                obj_center, obj_pcd, obj_mask = perception_result
                                print(f"感知成功! 找到物体中心: {obj_center}")
                                print(f"获取到物体点云，包含 {len(obj_pcd)} 个点。")
                                
                                if len(obj_pcd) > 100: # 只有点数足够才替换
                                    masked_point_cloud_np = obj_pcd
                                    print(">>> 将使用 Mask 过滤后的点云进行 GraspNet 推理 <<<")
                                else:
                                    print("警告: 物体点云点数过少，回退到使用全场景点云。")
                            else:
                                 print(f"感知管道返回格式不匹配 (期望 3 个值，得到 {len(perception_result)} 个)。使用全场景点云。")
                        else:
                            print("感知管道未能检测到物体。使用全场景点云。")
                            
                    except Exception as perc_err:
                        print(f"感知管道运行出错: {perc_err}")
                        import traceback; traceback.print_exc()
                        print("将继续使用全场景点云。")
            
            # --- 感知管道结束 ---

            # --- 调用 GraspNet 推理 --- 
            predicted_grasps_cam_frame = []
            graspgroup_result = None  # 用于存储 GraspGroup 对象（用于可视化）
            
            # 只有当 RGB 文件存在、点云生成成功、内参获取成功时才调用
            if final_rgb_exists and masked_point_cloud_np is not None and intrinsics_exist:
                try:
                    graspnet_checkpoint = str(_project_root / "checkpoints/graspnet/checkpoint-rs.tar") # 您的检查点路径
                    
                    # 检查 checkpoint 是否存在，如果不存在尝试其他常见路径
                    if not os.path.exists(graspnet_checkpoint):
                        # 尝试 manipulator_grasp 目录
                        alt_ckpt = os.path.join(os.path.dirname(_project_root), "manipulator_grasp", "checkpoints", "graspnet", "checkpoint-rs.tar")
                        if os.path.exists(alt_ckpt):
                            graspnet_checkpoint = alt_ckpt
                            print(f"在备用路径找到 GraspNet 权重: {graspnet_checkpoint}")
                    
                    # 调用 GraspNet（传入 MASKED 点云）
                    result = get_grasp_poses_from_graspnet(
                        rgb_image_path=final_image_filename, 
                        points_np=masked_point_cloud_np, # <--- 使用过滤后的点云
                        intrinsics_matrix=intrinsics_cv_grasp,
                        checkpoint_path=graspnet_checkpoint,
                        use_real_graspnet=True  # 设置为 True 启用真实 GraspNet
                    )
                    
                    # 检查返回类型
                    if GRASPNET_AVAILABLE and isinstance(result, GraspGroup):
                        graspgroup_result = result
                        print(f"成功从 GraspNet 获取 GraspGroup (包含 {len(result)} 个抓取位姿)。")
                        # 转换为矩阵列表供后续使用
                        predicted_grasps_cam_frame = graspgroup_to_matrices(result, max_grasps=10)
                    elif isinstance(result, list):
                        predicted_grasps_cam_frame = result
                        print(f"成功从 GraspNet 获取 {len(result)} 个抓取位姿 (相机坐标系)。")
                    else:
                        print("GraspNet 未返回有效抓取位姿。")
                        
                except Exception as gn_err:
                    print(f"调用 GraspNet 时出错: {gn_err}")
                    import traceback; traceback.print_exc()
            else:
                 print("缺少最终 RGB、点云或内参，无法调用 GraspNet。")
                 # 打印更详细的缺失信息
                 if not final_rgb_exists: print(f"  RGB file missing or path invalid: {final_image_filename}")
                 if masked_point_cloud_np is None: print("  Point cloud not generated.")
                 if not intrinsics_exist: print("  Intrinsics missing.")
            # --- GraspNet 调用结束 --- 
            
            # --- 新增: 调用可视化 ---
            # 优先使用 GraspGroup 对象（包含夹爪几何体），否则使用矩阵列表
            grasps_to_visualize = graspgroup_result if graspgroup_result is not None else predicted_grasps_cam_frame
            
            if grasps_to_visualize and (isinstance(grasps_to_visualize, list) and len(grasps_to_visualize) > 0 or 
                                        (GRASPNET_AVAILABLE and isinstance(grasps_to_visualize, GraspGroup) and len(grasps_to_visualize) > 0)):
                vis_filename = os.path.join(image_save_dir, "graspnet_vis.png")
                print(f"正在生成抓取可视化...")
                try:
                    # 将 masked_point_cloud_np 转换为 Open3D 点云
                    vis_pcd = o3d.geometry.PointCloud()
                    vis_pcd.points = o3d.utility.Vector3dVector(masked_point_cloud_np)
                    
                    # 传入 GraspGroup 或矩阵列表
                    # 使用更大的窗口尺寸便于观察
                    visualize_grasps(vis_pcd, grasps_to_visualize, intrinsics_cv_grasp, np.eye(4), vis_filename, width=800, height=600, max_grasps=10)
                except Exception as vis_err:
                    import traceback
                    print(f"可视化生成失败: {vis_err}")
                    print(traceback.format_exc())
            else:
                if masked_point_cloud_np is None:
                    print("警告: 点云为空，跳过可视化。")
                if not grasps_to_visualize:
                    print("警告: 没有预测的抓取位姿，跳过可视化。")
            # --- 可视化结束 ---

            # --- 选择抓取位姿并转换到基座标系 --- 
            target_grasp_pose_base_sapien = None # 初始化最终用于 IK 的目标位姿 (sapien.Pose)
            chosen_grasp_T_Base_Grasp = None   # 初始化变换后的矩阵
            T_Base_Cam_current = None          # 初始化当前 T_Base_Cam
            
            if predicted_grasps_cam_frame: # 仅当 GraspNet 返回结果时处理
                print("\\n--- 处理抓取位姿 ---")
                # 1. 选择一个抓取位姿 (简单选择第一个)
                chosen_grasp_T_Cam_Grasp = predicted_grasps_cam_frame[0]
                print(f"选择的抓取位姿 T_Cam_Grasp:\\n{np.round(chosen_grasp_T_Cam_Grasp, 3)}")

                # 2. 计算当前相机到基座的变换 T_Base_Cam
                print("计算当前的 T_Base_Cam...")
                T_Base_Cam_current = get_camera_pose_in_base(agent_left, camera_link_name="camera_link")
                
                # 3. 转换抓取位姿到基座标系
                if T_Base_Cam_current is not None:
                    chosen_grasp_T_Base_Grasp = transform_grasp_to_base(
                        grasp_pose_cam=chosen_grasp_T_Cam_Grasp, 
                        T_Base_Cam=T_Base_Cam_current
                    )
                else:
                    print("错误: 未能计算 T_Base_Cam，无法转换抓取位姿。")
                    
                # 4. 将基座标系抓取位姿矩阵转换为 sapien.Pose
                if chosen_grasp_T_Base_Grasp is not None:
                    target_grasp_pose_base_sapien = matrix_to_sapien_pose(chosen_grasp_T_Base_Grasp)
                    if target_grasp_pose_base_sapien is not None:
                        print("成功将抓取位姿转换为 sapien.Pose 对象。")
                        
                        # --- 新增: 在 SAPIEN 场景中绘制抓取位姿 ---
                        try:
                            # 获取基座的世界位姿 T_World_Base
                            base_pose = agent_left.robot.pose
                            p_base = base_pose.p
                            q_base = base_pose.q
                            if hasattr(p_base, 'cpu'): p_base = p_base.cpu().numpy()
                            if hasattr(q_base, 'cpu'): q_base = q_base.cpu().numpy()
                            T_World_Base = pose_to_transformation_matrix(p_base, q_base)
                            
                            # 计算抓取的世界位姿 T_World_Grasp = T_World_Base * T_Base_Grasp
                            T_World_Grasp = T_World_Base @ chosen_grasp_T_Base_Grasp
                            
                            # 提取位置和四元数
                            grasp_p_world = T_World_Grasp[:3, 3]
                            grasp_q_world_wxyz = tq.mat2quat(T_World_Grasp[:3, :3])
                            grasp_pose_world = sapien.Pose(p=grasp_p_world, q=grasp_q_world_wxyz)
                            
                            # 临时注释掉 SAPIEN 标记绘制，因为版本兼容性问题导致 crash
                            # try:
                            #     print(f"在 SAPIEN 场景中添加抓取标记: {grasp_p_world}")
                            #     builder = env.scene.create_actor_builder()
                            #     
                            #     # X轴 (红色)
                            #     builder.add_box_visual(half_size=[0.05, 0.002, 0.002], material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]), pose=sapien.Pose(p=[0.05, 0, 0]))
                            #     # Y轴 (绿色)
                            #     builder.add_box_visual(half_size=[0.002, 0.05, 0.002], material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]), pose=sapien.Pose(p=[0, 0.05, 0]))
                            #     # Z轴 (蓝色)
                            #     builder.add_box_visual(half_size=[0.002, 0.002, 0.05], material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]), pose=sapien.Pose(p=[0, 0, 0.05]))
                            #     
                            #     grasp_marker = builder.build_static(name="grasp_pose_marker")
                            #     grasp_marker.set_pose(grasp_pose_world)
                            #     
                            # except Exception as marker_err:
                            #     print(f"在 SAPIEN 中添加抓取标记时出错: {marker_err}")
                        except Exception as e:
                            print(f"计算或显示抓取标记时出错: {e}")
                        # --- 新增结束 ---
                        # --- 新增结束 ---
                        
                    else:
                        print("错误: 转换抓取位姿矩阵到 sapien.Pose 失败。")
                else:
                    print("没有有效的基座标系抓取位姿矩阵可供转换。")
                    
            # --- 抓取位姿处理结束 ---
            
            # --- 基于 GraspNet 结果进行 IK 和移动 (重构为分阶段) --- 
            if target_grasp_pose_base_sapien is not None: # 仅当成功获得目标 Sapien Pose 时执行
                print("\\n" + "-"*15 + " 开始分阶段抓取执行 " + "-"*15)
                
                # 通用 IK 参数
                urdf_path_for_ik = str(_project_root / 'urdf_01/GEN72.urdf')
                ee_link_name = agent_left.ee_link_name
                robot_articulation = agent_left.robot
                all_sapien_joint_names = [j.get_name() for j in robot_articulation.get_active_joints()]
                
                # ===================================================================
                # 阶段 2.1: Pre-Grasp (移动到物体上方 20cm)
                # ===================================================================
                print("\\n>>> 阶段 2.1: Pre-Grasp (移动到物体上方)...")
                pre_grasp_height = 0.20 # 20 cm
                # 简单沿 Z 轴抬高
                pre_grasp_pos = target_grasp_pose_base_sapien.p + np.array([0, 0, pre_grasp_height])
                
                # 保持抓取姿态
                pre_grasp_pose = sapien.Pose(p=pre_grasp_pos, q=target_grasp_pose_base_sapien.q)
                
                print(f"  Pre-Grasp 目标位置: {pre_grasp_pos}")
                
                # IK for Pre-Grasp
                current_qpos = agent_left.robot.get_qpos()
                ik_result_pre, ik_success_pre = compute_ik_with_pybullet(
                    urdf_path=urdf_path_for_ik,
                    ee_link_name=ee_link_name,
                    target_pose_base=pre_grasp_pose,
                    current_sapien_qpos=current_qpos,
                    all_sapien_joint_names=all_sapien_joint_names,
                    device=device, dtype=dtype
                )
                
                if ik_success_pre and ik_result_pre is not None:
                    print("Pre-Grasp IK 成功，开始平滑移动...")
                    # 移动，夹爪保持张开 (0.04)
                    gripper_open_val = 0.04 
                    smooth_move_to_qpos(env, ik_result_pre, gripper_open_val, num_steps=60)
                    print("Pre-Grasp 移动完成。")
                    time.sleep(0.5) 
                else:
                    print("错误: Pre-Grasp IK 失败，停止执行。")
                    # return 

                # ===================================================================
                # 阶段 2.2: Approach (垂直下降到抓取位姿)
                # ===================================================================
                print("\\n>>> 阶段 2.2: Approach (下降到抓取点)...")
                # 目标即为 target_grasp_pose_base_sapien
                
                # IK for Approach (使用 Pre-Grasp 后的 qpos 作为初值)
                current_qpos = agent_left.robot.get_qpos()
                ik_result_grasp, ik_success_grasp = compute_ik_with_pybullet(
                    urdf_path=urdf_path_for_ik,
                    ee_link_name=ee_link_name,
                    target_pose_base=target_grasp_pose_base_sapien,
                    current_sapien_qpos=current_qpos,
                    all_sapien_joint_names=all_sapien_joint_names,
                    device=device, dtype=dtype
                )
                
                if ik_success_grasp and ik_result_grasp is not None:
                    print("Approach IK 成功，开始平滑下降...")
                    # 移动，夹爪保持张开
                    smooth_move_to_qpos(env, ik_result_grasp, gripper_open_val, num_steps=40)
                    print("Approach 移动完成。")
                    time.sleep(0.2)
                else:
                    print("错误: Approach IK 失败，无法抓取。")

                # ===================================================================
                # 阶段 2.3: Close Gripper (闭合夹爪)
                # ===================================================================
                print("\\n>>> 阶段 2.3: Close Gripper (闭合夹爪)...")
                # 保持当前手臂位置，只改变夹爪值为 -0.01 (闭合)
                current_arm_qpos = agent_left.robot.get_qpos()[0, :7]
                smooth_move_to_qpos(env, current_arm_qpos, -0.01, num_steps=30)
                print("夹爪闭合完成。")
                time.sleep(0.5) 

                # ===================================================================
                # 阶段 2.4: Lift (抬起物体)
                # ===================================================================
                print("\\n>>> 阶段 2.4: Lift (抬起物体)...")
                lift_height = 0.20 # 抬起 20 cm
                lift_pos = target_grasp_pose_base_sapien.p + np.array([0, 0, lift_height])
                lift_pose = sapien.Pose(p=lift_pos, q=target_grasp_pose_base_sapien.q)
                
                # IK for Lift
                current_qpos = agent_left.robot.get_qpos()
                ik_result_lift, ik_success_lift = compute_ik_with_pybullet(
                    urdf_path=urdf_path_for_ik,
                    ee_link_name=ee_link_name,
                    target_pose_base=lift_pose,
                    current_sapien_qpos=current_qpos,
                    all_sapien_joint_names=all_sapien_joint_names,
                    device=device, dtype=dtype
                )
                
                if ik_success_lift and ik_result_lift is not None:
                    print("Lift IK 成功，开始平滑抬起...")
                    # 移动，夹爪保持闭合 (-0.01)
                    smooth_move_to_qpos(env, ik_result_lift, -0.01, num_steps=60)
                    print("抬起动作完成。")
                else:
                    print("错误: Lift IK 失败。")

            else:
                 print("\\n未能获取有效的 GraspNet 抓取位姿 (sapien.Pose)，跳过后续 IK 和移动步骤。")
            # --- 分阶段抓取结束 ---

        else:
             print("感知流程失败或被禁用，跳过第一次移动和后续 GraspNet 步骤。") 

        # --- 清理: 移除或注释掉基于 center_point_world 的移动代码 --- 
        # if center_point_world is not None: # <--- 这个 if 块现在应该被移除或注释掉
            # print("\\n--- 开始移动到目标点上方并拍照 --- ")
            # original_qpos = agent_left.robot.get_qpos().clone() # 保存原始 qpos 以便恢复
            # try:
            #     # ... (计算 target_pose_world, target_pose_base)
            #     # ... (准备 IK 输入)
            #     # ... (调用 compute_ik_with_pybullet)
            #     # ... (构造 8D 动作 action)
            #     # ... (循环调用 env.step(action))
            #     # ... (拍照)
            # except Exception as move_err:
            #     # ...
        # --- 清理结束 --- 
        
        # --- 添加最终的持续渲染循环 --- 
        print("\\n" + "="*30 + " 所有阶段完成，进入持续渲染循环 " + "="*30)
        print("按 Ctrl+C 退出。")
        try:
            while True:
                env.render()
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\\n接收到 Ctrl+C，退出渲染循环。")
        # --- 渲染循环结束 ---

    except Exception as e:
        import traceback
        print(f"处理相机数据或感知流程时出错: {e}")
        print(traceback.format_exc())
    # --- 拍照结束 -> 感知流程结束 ---

    finally:
        print("--- 最终调试: 再次打印香蕉真实世界坐标 --- ")
        try:
            # 确保 env 和 banana 属性/对象都有效
            if 'env' in locals() and env is not None and hasattr(env, 'banana') and env.banana is not None and hasattr(env.banana, 'pose'):
                 banana_pose_tensor = env.banana.pose.p
                 # 检查是否是 Tensor 并移到 CPU 转 NumPy
                 if hasattr(banana_pose_tensor, 'cpu'): 
                     banana_pose_np = banana_pose_tensor[0].cpu().numpy() 
                 else: # 假设已经是 NumPy 或类似结构
                     banana_pose_np = np.asarray(banana_pose_tensor)[0]
                 print(f"Banana Pose (Ground Truth): {banana_pose_np}")
            else:
                 print("无法获取 Banana 对象、其 Pose 或环境对象无效。")
        except Exception as e:
            print(f"--- 调试: 获取最终 Banana Pose 时出错: {e} ---")
        
        print("关闭环境...")
    env.close()

    # --- 新增结束 ---


    # 保持环境打开直到用户关闭
    # print("\\n环境设置完成。按 Ctrl+C 或关闭窗口退出。") # <<< 这行也被注释掉或移除 >>>

if __name__ == "__main__":
    test_set_pose_and_capture()
