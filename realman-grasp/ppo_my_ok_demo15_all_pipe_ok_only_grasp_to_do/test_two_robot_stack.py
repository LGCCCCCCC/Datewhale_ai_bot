#!/usr/bin/env python3
# ==============================================================================
#  IMPORTS & SETUP
# ==============================================================================
"""
测试双臂堆叠立方体环境 - 修改为设置姿态、拍照并保持
"""

import numpy as np
import gymnasium as gym
import time
import sys
import os
import cv2 # 添加cv2库
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

# --- 导入自定义工具和模块 ---
from grasp_prediction_utils import (
    depth_to_point_cloud,
    get_grasp_poses_from_graspnet,
    get_camera_pose_in_base,
    transform_grasp_to_base,
    matrix_to_sapien_pose,
    pose_to_transformation_matrix
)
# from perception_pipeline import run_perception # 取消注释当 perception_pipeline.py 创建后

# --- 路径设置 ---
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

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
#  Pose to Transformation Matrix
# ------------------------------------------------------------------------------
def pose_to_transformation_matrix(position, quaternion_wxyz):
    """将 (位置, 四元数 WXYZ) 转换为 4x4 变换矩阵"""
    mat = tq.quat2mat(quaternion_wxyz)
    transform = np.identity(4)
    transform[:3, :3] = mat
    transform[:3, 3] = position
    return transform

# ------------------------------------------------------------------------------
#  PyBullet Inverse Kinematics (IK)
# ------------------------------------------------------------------------------
def compute_ik_with_pybullet(
    urdf_path: str,
    ee_link_name: str,
    target_pose_base: sapien.Pose,
    current_sapien_qpos: torch.Tensor,
    all_sapien_joint_names: list[str],
    device: torch.device,
    dtype: torch.dtype
):
    """使用 PyBullet 计算逆运动学"""
    print("--- 开始执行 IK (PyBullet) --- ")
    physics_client = None
    try:
        # 1. 连接到 PyBullet
        physics_client = p.connect(p.DIRECT)
        if physics_client < 0:
             print("错误: 连接 PyBullet 失败!")
             return None, False
        
        print(f"尝试加载 URDF: {urdf_path}")
        p.setTimeStep(1./240.)
        robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        print(f"PyBullet 加载 URDF 成功, Robot ID: {robot_id}")

        # 2. 构建 SAPIEN 名称到 PyBullet 索引和信息的映射
        num_joints_pb = p.getNumJoints(robot_id)
        print(f"PyBullet 找到的总关节数: {num_joints_pb}")
        
        sapien_name_to_pb_index = {}
        pb_joint_indices_active = []
        pb_lower_limits = []
        pb_upper_limits = []
        pb_joint_ranges = []
        pb_rest_poses = []
        ee_link_pb_index = -1
        
        normalized_sapien_active_names_set = {name.strip() for name in all_sapien_joint_names}

        print("构建关节映射 (SAPIEN -> PyBullet)...")
        for i in range(num_joints_pb):
            joint_info = p.getJointInfo(robot_id, i)
            pb_joint_index = joint_info[0]
            pb_joint_name = joint_info[1].decode('utf-8').strip()
            pb_joint_type = joint_info[2]
            pb_link_name = joint_info[12].decode('utf-8').strip()
            pb_ll = joint_info[8]
            pb_ul = joint_info[9]

            # print(f"  PB Joint Index {pb_joint_index}: Name='{pb_joint_name}', Type={pb_joint_type}, Link='{pb_link_name}', Limits=({pb_ll:.4f}, {pb_ul:.4f})", end="")

            if pb_joint_name in normalized_sapien_active_names_set and pb_joint_type != p.JOINT_FIXED:
                sapien_name_to_pb_index[pb_joint_name] = pb_joint_index
                pb_joint_indices_active.append(pb_joint_index)
                pb_lower_limits.append(pb_ll)
                pb_upper_limits.append(pb_ul)
                pb_joint_ranges.append(pb_ul - pb_ll)
                pb_rest_poses.append((pb_ll + pb_ul) / 2.0)
                # print(" -> 活动关节 (匹配 SAPIEN)")
            # else:
                 # print(" -> 非活动或固定关节")

            if pb_link_name == ee_link_name:
                ee_link_pb_index = pb_joint_index
                # print(f"    -> 找到末端执行器 Link '{ee_link_name}' 对应的 PyBullet Joint Index: {ee_link_pb_index}")

        if ee_link_pb_index == -1:
            print(f"错误: 未能在 PyBullet 中找到名为 '{ee_link_name}' 的末端执行器 Link 对应的 Joint Index。")
            return None, False

        if len(pb_joint_indices_active) != len(all_sapien_joint_names):
             print(f"警告: PyBullet 中找到的可动关节数 ({len(pb_joint_indices_active)}) 与 SAPIEN 活动关节数 ({len(all_sapien_joint_names)}) 不匹配！")
             
        # print(f"SAPIEN 名称到 PyBullet 索引的映射: {sapien_name_to_pb_index}")
        # print(f"用于 IK 的 PyBullet 活动关节索引列表: {pb_joint_indices_active}")

        # 3. 准备 IK 输入
        target_pos = target_pose_base.p
        target_orn_wxyz = target_pose_base.q
        target_orn_xyzw = [target_orn_wxyz[1], target_orn_wxyz[2], target_orn_wxyz[3], target_orn_wxyz[0]]
        
        # print(f"目标位置 (基座标系): {target_pos}")
        # print(f"目标姿态 (基座标系 quat xyzw): {target_orn_xyzw}")

        current_sapien_qpos_np = current_sapien_qpos.squeeze(0).cpu().numpy()
        sapien_idx_map = {name: idx for idx, name in enumerate(all_sapien_joint_names)}
        
        # print("映射 SAPIEN qpos 到 PyBullet 初始猜测...")
        num_active_pb_joints = len(pb_joint_indices_active)
        if len(pb_lower_limits) != num_active_pb_joints: print("错误: 限制数量与活动关节数不匹配!")
        
        initial_guess_pb = [0.0] * num_active_pb_joints
        map_success = True
        for i, pb_idx in enumerate(pb_joint_indices_active):
             joint_info = p.getJointInfo(robot_id, pb_idx)
             pb_joint_name = joint_info[1].decode('utf-8').strip()
             if pb_joint_name in sapien_idx_map:
                  sapien_idx = sapien_idx_map[pb_joint_name]
                  try:
                      initial_guess_pb[i] = current_sapien_qpos_np[sapien_idx]
                  except IndexError:
                      print(f"错误: SAPIEN 索引 {sapien_idx} 超出 qpos 范围 for joint '{pb_joint_name}'")
                      map_success = False; break
             else:
                  print(f"错误: PyBullet 活动关节 '{pb_joint_name}' 未在 SAPIEN 索引映射中找到!")
                  map_success = False; break
        if not map_success:
             print("映射 SAPIEN qpos 到 PyBullet 失败。")
             return None, False
        # print(f"PyBullet 初始关节猜测 (len {len(initial_guess_pb)}): {np.round(initial_guess_pb, 4)}")

        # 4. 调用 PyBullet IK
        # print("调用 PyBullet calculateInverseKinematics (带限制，无初始猜测)...")
        ik_solution_pb = p.calculateInverseKinematics(
            robot_id,
            ee_link_pb_index,
            target_pos,
            target_orn_xyzw,
            lowerLimits=pb_lower_limits,
            upperLimits=pb_upper_limits,
            jointRanges=pb_joint_ranges,
            restPoses=pb_rest_poses,
        )
        
        if ik_solution_pb is None or len(ik_solution_pb) != num_active_pb_joints:
             print(f"错误: PyBullet IK 计算失败或返回结果长度不匹配。")
             # print("再次调用 IK (参数与第一次相同)...")
             ik_solution_pb = p.calculateInverseKinematics(
                 robot_id, ee_link_pb_index, target_pos, target_orn_xyzw,
                 lowerLimits=pb_lower_limits,
                 upperLimits=pb_upper_limits,
                 jointRanges=pb_joint_ranges,
                 restPoses=pb_rest_poses
             )
             if ik_solution_pb is None or len(ik_solution_pb) != num_active_pb_joints:
                  print("再次尝试 IK 失败。")
                  return None, False
             # else:
                  # print("不使用 currentPosition 的 IK 成功。")
                     
        # print(f"PyBullet IK 求解完成。解 (len {len(ik_solution_pb)}): {np.round(ik_solution_pb, 4)}")

        # 5. 将结果映射回 SAPIEN 格式 (返回 7D 结果)
        ik_qpos_sapien_7d_np = np.zeros(7)
        mapped_back_count = 0
        # print("准备将 IK 解映射回 SAPIEN 7D qpos (按 SAPIEN 前 7 活动关节顺序)...")

        pb_idx_to_solution_index = {pb_idx: i for i, pb_idx in enumerate(pb_joint_indices_active)}

        for sapien_idx in range(7):
            sapien_joint_name = all_sapien_joint_names[sapien_idx]
            if sapien_joint_name in sapien_name_to_pb_index:
                pb_idx = sapien_name_to_pb_index[sapien_joint_name]
                if pb_idx in pb_idx_to_solution_index:
                    solution_idx = pb_idx_to_solution_index[pb_idx]
                    ik_angle = ik_solution_pb[solution_idx]
                    ik_qpos_sapien_7d_np[sapien_idx] = ik_angle
                    mapped_back_count += 1
                else:
                    print(f"警告: PyBullet 索引 {pb_idx} 未在 IK 解索引映射中找到!")
            else:
                print(f"警告: SAPIEN 关节 '{sapien_joint_name}' 未在 PyBullet 名称映射中找到!")

        # print(f"成功将 {mapped_back_count} 个关节角度映射到 7D 结果。")
        if mapped_back_count != 7:
            print("警告: 未能精确映射 7 个关节! IK 结果可能不准确。")

        ik_qpos_result_7d = torch.tensor(ik_qpos_sapien_7d_np, dtype=dtype, device=device).unsqueeze(0)
        print("IK 计算成功完成 (返回 7D 结果)。")
        return ik_qpos_result_7d, True

    except Exception as pybullet_err:
        print(f"使用 PyBullet 进行 IK 计算时出错: {pybullet_err}")
        import traceback; traceback.print_exc()
        return None, False
    finally:
        if physics_client is not None and p.isConnected(physics_client):
            # print("断开 PyBullet 连接...")
            p.disconnect(physics_client)

# ------------------------------------------------------------------------------
#  Gripper Control Functions
# ------------------------------------------------------------------------------
# Gripper joint values for GEN72EG2Robot ('4C2_Joint1')
# These values are based on the robot's URDF and controller configuration.
# Lower limit for '4C2_Joint1' is typically 0.0 (closed).
# Upper limit for '4C2_Joint1' is typically 0.82 (fully open).
GRIPPER_OPEN_VALUE = 0.82
GRIPPER_CLOSE_VALUE = 0.0

def control_gripper(agent_to_control, env, gripper_target_qpos_value, num_render_steps=5):
    """
    Controls the gripper of a specified agent to a target qpos value
    by directly setting the robot's qpos.

    Args:
        agent_to_control: The agent whose gripper is to be controlled.
        env: The ManiSkill environment instance.
        gripper_target_qpos_value (float): The target qpos value for the gripper's driving joint.
        num_render_steps (int): Number of times to call scene.step() and render() after setting qpos,
                                for visual smoothness or minor simulation settling.
    """
    robot_articulation = agent_to_control.robot
    active_joints = robot_articulation.get_active_joints()
    
    # Determine the name of the gripper's main driving joint.
    # For GEN72EG2Robot, the gripper_joint_names = ['4C2_Joint1']
    # We assume the first (and typically only) joint in agent.gripper_joint_names is the one to control.
    # This is more robust than assuming a fixed index like 7.
    gripper_drive_joint_name = None
    if hasattr(agent_to_control, 'gripper_joint_names') and agent_to_control.gripper_joint_names:
        gripper_drive_joint_name = agent_to_control.gripper_joint_names[0]
    else:
        # Fallback for safety, though GEN72EG2Robot should have gripper_joint_names
        print(f"Warning: agent {agent_to_control.uid} does not have 'gripper_joint_names' defined. Falling back to '4C2_Joint1'.")
        gripper_drive_joint_name = '4C2_Joint1'

    gripper_joint_idx = -1
    for idx, joint in enumerate(active_joints):
        if joint.name == gripper_drive_joint_name:
            gripper_joint_idx = idx
            break
    
    if gripper_joint_idx == -1:
        print(f"Error: Could not find gripper drive joint '{gripper_drive_joint_name}' for agent {agent_to_control.uid}. \nAvailable active joints: {[j.name for j in active_joints]}")
        return

    current_qpos_full = robot_articulation.get_qpos() # Shape [1, N_active_joints]
    if not (current_qpos_full.ndim == 2 and current_qpos_full.shape[0] == 1):
        print(f"Error: Unexpected qpos shape {current_qpos_full.shape} for agent {agent_to_control.uid}. Expected [1, N].")
        return
    if current_qpos_full.shape[1] <= gripper_joint_idx:
        print(f"Error: qpos length {current_qpos_full.shape[1]} for agent {agent_to_control.uid} \n is not sufficient for gripper_joint_idx {gripper_joint_idx} ('{gripper_drive_joint_name}').")
        return
        
    modified_qpos_full = current_qpos_full.clone()
    original_gripper_qpos = modified_qpos_full[0, gripper_joint_idx].item()
    modified_qpos_full[0, gripper_joint_idx] = gripper_target_qpos_value
    
    agent_description = "Unknown"
    all_agents = env.agent.agents
    if agent_to_control is all_agents[0]: 
        agent_description = "Left Agent (agent_0)"
    elif len(all_agents) > 1 and agent_to_control is all_agents[1]: 
        agent_description = "Right Agent (agent_1)"

    print(f"Directly setting qpos for gripper of {agent_description} (joint: '{gripper_drive_joint_name}', index: {gripper_joint_idx}).")
    print(f"  Original value: {original_gripper_qpos:.4f}, Target value: {gripper_target_qpos_value:.4f}")
    
    robot_articulation.set_qpos(modified_qpos_full)
    
    # Step the scene and render a few times
    for i in range(max(1, num_render_steps)): # Ensure at least one step
        env.scene.step() 
        if env.render_mode == "human":
            env.render()
            if num_render_steps > 1: # Only sleep if multiple render steps are intended for visuals
                 time.sleep(0.02) 
    
    final_gripper_qpos = robot_articulation.get_qpos()[0, gripper_joint_idx].item()
    print(f"Gripper control via set_qpos for {agent_description} complete. Final value: {final_gripper_qpos:.4f}")

def open_gripper(agent_to_control, env, num_render_steps=5):
    """Opens the gripper of the specified agent by directly setting qpos."""
    agent_name = "Left" if agent_to_control == env.agent.agents[0] else "Right" if (len(env.agent.agents) > 1 and agent_to_control == env.agent.agents[1]) else "Unknown"
    print(f"--- Attempting to OPEN gripper for {agent_name} agent (using set_qpos) ---")
    control_gripper(agent_to_control, env, GRIPPER_OPEN_VALUE, num_render_steps)

def close_gripper(agent_to_control, env, num_render_steps=5):
    """Closes the gripper of the specified agent by directly setting qpos."""
    agent_name = "Left" if agent_to_control == env.agent.agents[0] else "Right" if (len(env.agent.agents) > 1 and agent_to_control == env.agent.agents[1]) else "Unknown"
    print(f"--- Attempting to CLOSE gripper for {agent_name} agent (using set_qpos) ---")
    control_gripper(agent_to_control, env, GRIPPER_CLOSE_VALUE, num_render_steps)

# ==============================================================================
#  MAIN TEST FUNCTION: test_set_pose_and_capture
# ==============================================================================
def test_set_pose_and_capture():
    """设置左臂姿态，拍照并保持渲染"""
    
    # --------------------------------------------------------------------------
    #  1. ENVIRONMENT CREATION & RESET
    # --------------------------------------------------------------------------
    print("="*20 + " 1. ENVIRONMENT CREATION & RESET " + "="*20)
    print("创建环境...")
    controller_choice = 'pd_joint_pos'
    print(f"指定环境使用控制器配置: {controller_choice}")
    env = TwoRobotStackCubeEnv(
        render_mode="human", 
        shader_dir="rt-fast", 
        obs_mode="rgb+depth",
        control_mode=controller_choice
    )
    
    print("重置环境...")
    try:
        obs, info = env.reset()
        print("使用新版Gymnasium API (返回obs, info)")
    except ValueError:
        obs = env.reset()
        print("使用旧版API (仅返回obs)")

    # --------------------------------------------------------------------------
    #  2. INITIAL LEFT ARM POSING
    # --------------------------------------------------------------------------
    print("\n" + "="*20 + " 2. INITIAL LEFT ARM POSING " + "="*20)
    print("手动设置左臂初始姿态...")
    try:
        left_agent = env.agent.agents[0]
        qpos = left_agent.robot.get_qpos()

        elbow_joint_index = 5
        angle_offset = -70.0 * torch.pi / 180.0
        angle_offset_tensor = torch.tensor(angle_offset, device=qpos.device, dtype=qpos.dtype)
        
        if qpos.shape[0] == 1 and qpos.shape[1] > elbow_joint_index:
            current_elbow_qpos = qpos[0, elbow_joint_index]
            new_elbow_qpos = current_elbow_qpos + angle_offset_tensor
            
            qlimits = left_agent.robot.get_qlimits()
            lower_limit = qlimits[0, elbow_joint_index, 0]
            upper_limit = qlimits[0, elbow_joint_index, 1]
            
            new_elbow_qpos_clamped = torch.clamp(new_elbow_qpos, lower_limit, upper_limit)
            qpos[0, elbow_joint_index] = new_elbow_qpos_clamped
            
            left_agent.robot.set_qpos(qpos)
            env.scene.step()
            
            print(f"左臂关节 {elbow_joint_index} 已设置为目标角度 (限制在范围内)。")
            
            try:
                banana_z = env.banana.pose.p[0, 2] if env.banana else 'N/A'
                table_z = env.table_scene.table.pose.p[0, 2] if hasattr(env, 'table_scene') and env.table_scene.table else 'N/A'
                print(f"--- 调试: 物体世界 Z 坐标 --- Banana Z: {banana_z}, Table Z: {table_z}")
            except Exception as e:
                print(f"--- 调试: 获取物体 Z 坐标时出错: {e} ---")
        else:
            print(f"错误：无法获取或设置左臂 qpos。Shape: {qpos.shape}, Index: {elbow_joint_index}")

    except Exception as e:
        import traceback
        print(f"设置左臂姿态时出错: {e}")
        print(traceback.format_exc())

    # --------------------------------------------------------------------------
    #  3. INITIAL IMAGE CAPTURE (RGB + DEPTH) & PARAMETER EXTRACTION
    # --------------------------------------------------------------------------
    print("\n" + "="*20 + " 3. INITIAL IMAGE CAPTURE & PARAMETERS " + "="*20)
    print("尝试从左臂腕部相机拍照...")
    image_save_dir = os.path.join(current_dir, "capture_images")
    os.makedirs(image_save_dir, exist_ok=True)
    image_filename = os.path.join(image_save_dir, f"left_arm_capture.png")
    depth_filename = os.path.join(image_save_dir, f"left_arm_depth.npy")
    camera_name = 'hand_camera' 
    agent_left = env.agent.agents[0]
    agent_id = "agent_0"

    rgb_saved = False
    depth_saved = False
    camera_name_base = "hand_camera"

    try:
        if hasattr(agent_left, 'sensors') and camera_name_base in agent_left.sensors:
            camera = agent_left.sensors[camera_name_base]
            print(f"正在捕获机器人 {agent_id} 手部相机 '{camera_name_base}' 图像和参数...")
            
            env.render()
            camera.capture()
            obs_dict = camera.get_obs(rgb=True, depth=True, segmentation=False)

            if 'rgb' in obs_dict:
                rgb_image = obs_dict['rgb']
                rgb_image_np = rgb_image.cpu().numpy() if hasattr(rgb_image, 'cpu') else np.array(rgb_image)
                if rgb_image_np.ndim == 4 and rgb_image_np.shape[0] == 1: rgb_image_np = rgb_image_np[0]
                elif rgb_image_np.ndim != 3: raise ValueError(f"RGB 图像维度不正确: {rgb_image_np.shape}")
                bgr_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2BGR) if rgb_image_np.dtype == np.uint8 else cv2.cvtColor((np.clip(rgb_image_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_filename, bgr_image)
                print(f"左臂手部相机 RGB 图像已保存至: {image_filename}")
                rgb_saved = True
            else:
                print(f"错误: 未能从相机 '{camera_name_base}' 获取 'rgb' 数据。")

            if 'depth' in obs_dict:
                depth_image = obs_dict['depth']
                depth_image_np = depth_image.cpu().numpy() if hasattr(depth_image, 'cpu') else np.array(depth_image)
                if depth_image_np.ndim == 4 and depth_image_np.shape[0] == 1: depth_image_np = depth_image_np[0]
                elif depth_image_np.ndim == 2: depth_image_np = np.expand_dims(depth_image_np, axis=-1)
                elif depth_image_np.ndim != 3 or depth_image_np.shape[-1] != 1: raise ValueError(f"深度图像维度不正确: {depth_image_np.shape}")
                np.save(depth_filename, depth_image_np)
                print(f"左臂手部相机深度图像已保存至: {depth_filename} (格式: {depth_image_np.dtype}, 形状: {depth_image_np.shape}, 单位: 毫米)")
                depth_saved = True
            else:
                 print(f"错误: 未能从相机 '{camera_name_base}' 获取 'depth' 数据。")

            # --- 获取最新的相机参数 ---
            print("在拍照后获取最新的观测值 (包含相机参数)... ")
            current_obs = None
            try:
                current_obs = env.get_obs()
                print("成功调用 env.get_obs()")
            except AttributeError:
                print("警告: env.get_obs() 不可用，尝试调用 env.step() 获取新观测值。")
                try:
                     action_space = env.action_space
                     dummy_action = {k: torch.zeros_like(v.sample()) for k, v in action_space.spaces.items()} if isinstance(action_space, gym.spaces.Dict) else torch.zeros_like(torch.from_numpy(action_space.sample()))
                     current_obs, _, _, _, info = env.step(dummy_action) 
                     print("通过 env.step(dummy_action) 获取了新的 obs。")
                except Exception as step_err: print(f"尝试 env.step() 获取新 obs 失败: {step_err}"); current_obs = obs
            except Exception as get_obs_err: print(f"调用 env.get_obs() 出错: {get_obs_err}"); current_obs = obs

            intrinsics_cv = None
            cam2world_matrix = None
            extrinsic_cv = None

            print(f"尝试从最新的观测值中提取 '{camera_name_base}' 的参数...")
            try:
                if "sensor_param" in current_obs and isinstance(current_obs["sensor_param"], dict) and camera_name_base in current_obs["sensor_param"]:
                    sensor_params = current_obs["sensor_param"][camera_name_base]
                    if "intrinsic_cv" in sensor_params:
                        intrinsics_cv = sensor_params["intrinsic_cv"]
                        if hasattr(intrinsics_cv, 'cpu'): intrinsics_cv = intrinsics_cv.cpu().numpy()
                        if intrinsics_cv.ndim == 3 and intrinsics_cv.shape[0] == 1: intrinsics_cv = intrinsics_cv.squeeze(0)
                        print(f"从新 obs 获取到内参 (Shape: {intrinsics_cv.shape})")
                    else: print("警告: 新 obs 中未找到 'intrinsic_cv'")
                    
                    if "cam2world_gl" in sensor_params:
                        cam2world_matrix = sensor_params["cam2world_gl"]
                        if hasattr(cam2world_matrix, 'cpu'): cam2world_matrix = cam2world_matrix.cpu().numpy()
                        if cam2world_matrix.ndim == 3 and cam2world_matrix.shape[0] == 1: cam2world_matrix = cam2world_matrix.squeeze(0)
                    
                    if "extrinsic_cv" in sensor_params:
                        extrinsic_cv = sensor_params["extrinsic_cv"]
                        if hasattr(extrinsic_cv, 'cpu'): extrinsic_cv = extrinsic_cv.cpu().numpy()
                        if extrinsic_cv.ndim == 3 and extrinsic_cv.shape[0] == 1: extrinsic_cv = extrinsic_cv.squeeze(0)
                        print(f"从新 obs 获取到外参 extrinsic_cv (Shape: {extrinsic_cv.shape})")
                    else: print("警告: 新 obs 中未找到 'extrinsic_cv'"); extrinsic_cv = None

                    # +++ 调试: 验证相机位姿 +++
                    # print(f"\n--- 调试: 验证 cam2world_matrix ---")
                    # ... (调试代码已为简洁省略，可从原始文件恢复)
                    # print(f"--- 调试结束 ---\n")
                    
                    # +++ 调试: 获取 camera_link 世界位姿 +++
                    T_World_Link = None
                    # print("\n--- 调试: 尝试获取 camera_link 的世界位姿 ---")
                    # ... (调试代码已为简洁省略)
                    # print("--- 调试结束 ---\n")

                else:
                    print(f"警告: 未能在新的 obs['sensor_param'] 中找到相机 '{camera_name_base}'。")
            except Exception as e:
                print(f"从新 obs 获取相机参数时出错: {e}")

            # --------------------------------------------------------------------------
            #  4. PERCEPTION PIPELINE (Placeholder/Fixed Value)
            # --------------------------------------------------------------------------
            print("\n" + "="*20 + " 4. PERCEPTION PIPELINE " + "="*20)
            print("--- 检查感知流程前置条件 ---")
            # print(f"rgb_saved: {rgb_saved}, depth_saved: {depth_saved}")
            # print(f"intrinsics_cv is None: {intrinsics_cv is None}, cam2world_matrix is None: {cam2world_matrix is None}, extrinsic_cv is None: {extrinsic_cv is None}")
            # print("-------------------------------")
            
            center_point_world = None
            if rgb_saved and depth_saved and intrinsics_cv is not None and cam2world_matrix is not None:
                print("--- 感知流程 (当前使用固定值) ---")
                # run_perception = None # 强制禁用
                # center_point_camera = None
                # if run_perception:
                    # ... (感知调用代码已为简洁省略)
                # else:
                    # print("错误: run_perception 未成功导入或被禁用。")

                # if center_point_camera is not None:
                    # ... (相机坐标系处理和验证代码已为简洁省略)
                    # --- 最终坐标变换 (当前跳过，使用固定值) ---
                    # center_point_world = ...
                # else:
                    # print("感知流程失败或被禁用，使用固定的世界坐标。")

                # --- 使用固定的世界坐标进行后续操作 ---
                center_point_world = np.array([0.48967573, 0.13179222, 0.03743867]) # Banana-like position
                print(f"使用固定的世界坐标 (用于后续移动): {center_point_world}")

                # --- 在场景中添加标记点 (可选) ---
                if center_point_world is not None:
                    print(f"尝试在场景中添加目标点标记 ({center_point_world})...")
                    try:
                        builder = env.scene.create_actor_builder()
                        render_material = sapien.render.RenderMaterial(base_color=[0, 1, 1, 1]) # 青色
                        builder.add_sphere_visual(radius=0.01, material=render_material)
                        marker_actor = builder.build_static(name="target_marker")
                        marker_actor.set_pose(sapien.Pose(p=center_point_world))
                        print(f"成功添加目标点标记 (Actor: {marker_actor.name})。")
                    except Exception as marker_err:
                        print(f"添加标记点时出错: {marker_err}")
            else:
                print("感知流程的必要参数不足 (RGB/Depth/Intrinsics/Extrinsics)，跳过。")

            # --------------------------------------------------------------------------
            #  5. FIRST IK MOVE: To Point Above Target (center_point_world)
            # --------------------------------------------------------------------------
            print("\n" + "="*20 + " 5. FIRST IK MOVE (ABOVE TARGET) " + "="*20)
            if center_point_world is not None:
                print("--- 开始移动到目标点上方并拍照 --- ")
                # original_qpos = agent_left.robot.get_qpos().clone() # 可用于恢复
                try:
                    # 1. 计算目标位姿 (世界和基座标系)
                    target_pos_world = center_point_world + np.array([0, 0, 0.3]) # 上方0.3m
                    target_rot_mat_world = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]) # 俯视
                    target_quat_world_xyzw = tq.mat2quat(target_rot_mat_world)
                    target_pose_world_sapien = sapien.Pose(p=target_pos_world, q=target_quat_world_xyzw)
                    # print(f"目标 EE 世界位姿: p={target_pos_world}, q_xyzw={target_quat_world_xyzw}")

                    base_pose_p = agent_left.robot.pose.p.cpu().numpy().squeeze(0)
                    base_pose_q = agent_left.robot.pose.q.cpu().numpy().squeeze(0)
                    T_World_Base = pose_to_transformation_matrix(base_pose_p, base_pose_q)
                    T_Base_World = np.linalg.inv(T_World_Base)
                    
                    T_World_TargetEE = pose_to_transformation_matrix(target_pose_world_sapien.p, target_pose_world_sapien.q)
                    T_Base_TargetEE = T_Base_World @ T_World_TargetEE
                    target_p_base = T_Base_TargetEE[:3, 3]
                    target_q_base_xyzw = tq.mat2quat(T_Base_TargetEE[:3, :3])
                    target_pose_base_sapien = sapien.Pose(p=target_p_base, q=target_q_base_xyzw)
                    # print(f"目标 EE 基座标系位姿: p={target_p_base}, q_xyzw={target_q_base_xyzw}")

                    # 2. 准备 IK 输入
                    urdf_path_for_ik = '/home/kewei/17robo/01mydemo/urdf_01/GEN72.urdf'
                    ee_link_name = agent_left.ee_link_name
                    current_sapien_qpos = agent_left.robot.get_qpos()
                    all_sapien_joint_names = [j.get_name() for j in agent_left.robot.get_active_joints()]
                    device = current_sapien_qpos.device
                    dtype = current_sapien_qpos.dtype
                    # print(f"准备调用 PyBullet IK: URDF='{urdf_path_for_ik}', EE='{ee_link_name}'")
                        
                    # 3. 调用 PyBullet IK
                    ik_qpos_7d, ik_success = compute_ik_with_pybullet(
                        urdf_path=urdf_path_for_ik, ee_link_name=ee_link_name,
                        target_pose_base=target_pose_base_sapien, current_sapien_qpos=current_sapien_qpos,
                        all_sapien_joint_names=all_sapien_joint_names, device=device, dtype=dtype
                    )
                        
                    if ik_success and ik_qpos_7d is not None:
                        print(f"PyBullet IK 成功找到 7D 解: {ik_qpos_7d.cpu().numpy().round(4)}")
                        # 4. 执行移动 (env.step)
                        print("构造 8D 动作并通过 env.step() 模拟...")
                        qpos_left_current = agent_left.robot.get_qpos()
                        qpos_right_current = env.agent.agents[1].robot.get_qpos()
                        gripper_joint_index = 7; num_controlled_joints = 8
                        action = None
                        try:
                            target_qpos_left_7d = ik_qpos_7d.squeeze(0)
                            current_gripper_left = qpos_left_current[0, gripper_joint_index:gripper_joint_index+1]
                            action_left_8d = torch.cat((target_qpos_left_7d, current_gripper_left), dim=0).to(device=device, dtype=dtype)
                            action_right_8d = qpos_right_current.squeeze(0)[:num_controlled_joints].to(device=device, dtype=dtype)
                            action_keys = list(env.action_space.spaces.keys())
                            action = {action_keys[0]: action_left_8d, action_keys[1]: action_right_8d}
                        except Exception as act_err: print(f"构造动作时出错: {act_err}")

                        if action is not None:
                            num_steps = 50
                            for step_idx in range(num_steps):
                                try:
                                    obs, reward, terminated, truncated, info = env.step(action)
                                    if env.render_mode == "human": env.render()
                                except Exception as step_err: print(f"env.step()出错: {step_err}"); break
                            print(f"已完成 {num_steps} 步模拟。")
                            
                            # 5. 拍照 (After Move)
                            print("执行移动后拍照...")
                            image_save_dir_after_move = os.path.join(current_dir, "capture_images_after_move")
                            os.makedirs(image_save_dir_after_move, exist_ok=True)
                            final_image_filename = os.path.join(image_save_dir_after_move, f"final_pose_capture.png")
                            final_depth_filename = os.path.join(image_save_dir_after_move, f"final_pose_depth.npy")

                            if hasattr(agent_left, 'sensors') and camera_name in agent_left.sensors:
                                camera = agent_left.sensors[camera_name]
                                env.render(); camera.capture()
                                obs_dict_final = camera.get_obs(rgb=True, depth=True)

                                if 'rgb' in obs_dict_final:
                                    rgb_final = obs_dict_final['rgb'].cpu().numpy()[0]
                                    bgr_final = cv2.cvtColor(rgb_final, cv2.COLOR_RGB2BGR) if rgb_final.dtype == np.uint8 else cv2.cvtColor((np.clip(rgb_final, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(final_image_filename, bgr_final)
                                    print(f"移动后 RGB 图像已保存至: {final_image_filename}")
                                else: print("错误: 未能获取移动后 RGB 图像。")

                                if 'depth' in obs_dict_final:
                                    depth_final = obs_dict_final['depth'].cpu().numpy()[0]
                                    if depth_final.ndim == 2: depth_final = np.expand_dims(depth_final, axis=-1)
                                    np.save(final_depth_filename, depth_final)
                                    print(f"移动后深度图像已保存至: {final_depth_filename}")
                                else: print("错误: 未能获取移动后深度图像。")
                            else: print(f"错误: 无法访问相机 '{camera_name}' 进行最终拍照。")
                        else: print("错误: 动作构建失败，跳过 env.step()。")
                    else: print("第一次 IK 失败，跳过移动和拍照。")
                except Exception as move_err:
                    print(f"在第一次移动和拍照过程中发生错误: {move_err}")
                    import traceback; traceback.print_exc()
            else:
                print("未能获取有效目标点 (center_point_world is None)，跳过第一次移动。")
            
            # --------------------------------------------------------------------------
            #  6. GRASPNET PREDICTION & SECOND IK MOVE (Placeholder)
            # --------------------------------------------------------------------------
            print("\n" + "="*20 + " 6. GRASPNET & SECOND IK MOVE (GRASP) " + "="*20)
            print("--- 第二阶段开始: GraspNet 预测与抓取尝试 ---")

            # --- 重新获取当前状态和相机参数 ---
            # print("准备 GraspNet: 重新获取当前状态和相机参数...")
            current_obs_grasp = None; intrinsics_cv_grasp = None
            try: # 获取最新观测值
                current_obs_grasp = env.get_obs()
            except: # 尝试 step(None) 或 dummy_action
                try: current_obs_grasp, _, _, _, _ = env.step(None)
                except: 
                    try: 
                        action_space = env.action_space
                        dummy_action = {k: torch.zeros_like(v.sample()) for k, v in action_space.spaces.items()} if isinstance(action_space, gym.spaces.Dict) else torch.zeros_like(torch.from_numpy(action_space.sample()))
                        current_obs_grasp, _, _, _, _ = env.step(dummy_action)
                    except: current_obs_grasp = current_obs # 退回
            
            if current_obs_grasp and "sensor_param" in current_obs_grasp and camera_name_base in current_obs_grasp["sensor_param"]:
                if "intrinsic_cv" in current_obs_grasp["sensor_param"][camera_name_base]:
                    intrinsics_cv_grasp = current_obs_grasp["sensor_param"][camera_name_base]["intrinsic_cv"]
                    if hasattr(intrinsics_cv_grasp, 'cpu'): intrinsics_cv_grasp = intrinsics_cv_grasp.cpu().numpy().squeeze(0)
                    # print(f"为 GraspNet 获取到内参 (Shape: {intrinsics_cv_grasp.shape})")

            # --- 点云生成 ---
            point_cloud_o3d = None; point_cloud_np = None
            final_rgb_exists = 'final_image_filename' in locals() and final_image_filename and os.path.exists(final_image_filename)
            final_depth_exists = 'final_depth_filename' in locals() and final_depth_filename and os.path.exists(final_depth_filename)
            intrinsics_exist_grasp = intrinsics_cv_grasp is not None
            
            if final_depth_exists and intrinsics_exist_grasp:
                # print("\n开始生成点云 (使用最终姿态深度图)...")
                point_cloud_o3d, point_cloud_np = depth_to_point_cloud(
                    depth_npy_path=final_depth_filename, intrinsics_matrix=intrinsics_cv_grasp,
                    depth_scale=1000.0, clip_distance_m=1.0
                )
                # if point_cloud_np is not None: print("点云生成成功。")
                # else: print("点云生成失败。")
            # else: print("缺少文件或内参，无法为 GraspNet 生成点云。")

            # --- GraspNet 推理 ---
            predicted_grasps_cam_frame = []
            if final_rgb_exists and point_cloud_np is not None and intrinsics_exist_grasp:
                try:
                    graspnet_checkpoint = "/home/kewei/17robo/graspnet-baseline/checkpoint-rs.tar"
                    predicted_grasps_cam_frame = get_grasp_poses_from_graspnet(
                        rgb_image_path=final_image_filename, points_np=point_cloud_np,
                        intrinsics_matrix=intrinsics_cv_grasp, checkpoint_path=graspnet_checkpoint
                    )
                    # if predicted_grasps_cam_frame: print(f"GraspNet 获取 {len(predicted_grasps_cam_frame)} 个抓取位姿。")
                    # else: print("GraspNet 未返回抓取位姿。")
                except Exception as gn_err: print(f"调用 GraspNet 时出错: {gn_err}")
            # else: print("缺少输入，无法调用 GraspNet。")
            
            # --- 选择抓取位姿并转换 ---
            target_grasp_pose_base_sapien = None
            if predicted_grasps_cam_frame:
                # print("\n处理抓取位姿...")
                chosen_grasp_T_Cam_Grasp = predicted_grasps_cam_frame[0] # 选择第一个
                T_Base_Cam_current = get_camera_pose_in_base(agent_left, camera_link_name="camera_link")
                if T_Base_Cam_current is not None:
                    chosen_grasp_T_Base_Grasp = transform_grasp_to_base(chosen_grasp_T_Cam_Grasp, T_Base_Cam_current)
                    target_grasp_pose_base_sapien = matrix_to_sapien_pose(chosen_grasp_T_Base_Grasp)
                    # if target_grasp_pose_base_sapien: print("成功将抓取位姿转换为 sapien.Pose。")
                # else: print("错误: 未能计算 T_Base_Cam。")
            
            # --- 基于 GraspNet 结果进行 IK 和移动 ---
            if target_grasp_pose_base_sapien is not None:
                print("\n" + "-"*10 + " 开始使用 GraspNet 位姿进行 IK 和移动 (抓取) " + "-"*10)
                # print("准备 IK 输入 (目标: GraspNet 位姿)...")
                current_sapien_qpos_grasp = agent_left.robot.get_qpos()
                
                ik_qpos_7d_grasp, ik_success_grasp = compute_ik_with_pybullet(
                    urdf_path=urdf_path_for_ik, ee_link_name=ee_link_name,
                    target_pose_base=target_grasp_pose_base_sapien, current_sapien_qpos=current_sapien_qpos_grasp,
                    all_sapien_joint_names=all_sapien_joint_names, device=device, dtype=dtype
                )
                
                if ik_success_grasp and ik_qpos_7d_grasp is not None:
                    print(f"为抓取位姿找到 IK 解: {ik_qpos_7d_grasp.cpu().numpy().round(4)}")
                    # print("构造 8D 动作并模拟移动到抓取位姿...")
                    qpos_right_current_grasp = env.agent.agents[1].robot.get_qpos()
                    action_grasp = None
                    try:
                        target_qpos_left_7d_grasp = ik_qpos_7d_grasp.squeeze(0)
                        qpos_left_current_grasp = agent_left.robot.get_qpos() 
                        current_gripper_left_grasp = qpos_left_current_grasp[0, gripper_joint_index:gripper_joint_index+1]
                        action_left_8d_grasp = torch.cat((target_qpos_left_7d_grasp, current_gripper_left_grasp), dim=0).to(device=device, dtype=dtype)
                        action_right_8d_grasp = qpos_right_current_grasp.squeeze(0)[:num_controlled_joints].to(device=device, dtype=dtype)
                        action_keys_grasp = list(env.action_space.spaces.keys())
                        action_grasp = {action_keys_grasp[0]: action_left_8d_grasp, action_keys_grasp[1]: action_right_8d_grasp}
                    except Exception as act_grasp_err: print(f"构造抓取动作时出错: {act_grasp_err}")
                    
                    if action_grasp is not None:
                        num_steps_grasp = 50
                        # print(f"开始执行 {num_steps_grasp} 步抓取模拟...")
                        for step_idx_g in range(num_steps_grasp):
                            try:
                                obs, reward, terminated, truncated, info = env.step(action_grasp)
                                if env.render_mode == "human": env.render()
                            except Exception as step_g_err: print(f"抓取 env.step()出错: {step_g_err}"); break
                        print(f"已完成 {num_steps_grasp} 步抓取模拟。")
                        # 可在此处添加抓取后拍照
                    else: print("错误: 抓取动作构建失败。")
                else: print("为抓取位姿进行的 IK 失败。")
            else:
                 print("\n未能获取有效的 GraspNet 抓取位姿，跳过后续抓取移动。")

            open_gripper(env.agent.agents[0], env)

            # --------------------------------------------------------------------------
            #  7. FINAL RENDERING LOOP
            # --------------------------------------------------------------------------
            print("\n" + "="*20 + " 7. FINAL RENDERING LOOP " + "="*20)
            print("所有阶段完成，进入持续渲染循环。按 Ctrl+C 退出。")
            try:
                while True:
                    env.render()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("\n接收到 Ctrl+C，退出渲染循环。")

        else: # agent_left.sensors or camera_name_base not found
             print(f"错误: 未找到相机 '{camera_name_base}' 或传感器列表。跳过后续流程。")

    except Exception as main_err:
        import traceback
        print(f"主流程中发生错误: {main_err}")
        print(traceback.format_exc())
    
    finally:
        # --------------------------------------------------------------------------
        #  8. CLEANUP & FINAL DEBUG INFO
        # --------------------------------------------------------------------------
        print("\n" + "="*20 + " 8. CLEANUP & FINAL INFO " + "="*20)
        print("--- 最终调试: 香蕉真实世界坐标 --- ")
        try:
            if 'env' in locals() and env is not None and hasattr(env, 'banana') and env.banana is not None:
                 banana_pose_tensor = env.banana.pose.p
                 banana_pose_np = banana_pose_tensor[0].cpu().numpy() if hasattr(banana_pose_tensor, 'cpu') else np.asarray(banana_pose_tensor)[0]
                 print(f"Banana Pose (Ground Truth): {banana_pose_np}")
            else: print("无法获取 Banana 对象或其 Pose。")
        except Exception as e: print(f"获取最终 Banana Pose 时出错: {e}")
        
        print("关闭环境...")
        if 'env' in locals() and env is not None:
            env.close()

# ==============================================================================
#  SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    test_set_pose_and_capture()