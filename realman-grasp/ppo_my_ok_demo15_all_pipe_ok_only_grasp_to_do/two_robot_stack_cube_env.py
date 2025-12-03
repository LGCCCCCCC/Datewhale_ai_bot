#!/usr/bin/env python3
"""
双臂堆叠立方体环境 - 基于ManiSkill框架使用GEN72-EG2机器人
"""

import os
import sys
import numpy as np
import torch
import sapien
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# 设置环境变量以使SAPIEN可以渲染
os.environ['SAPIEN_HEADLESS'] = '0'

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入GEN72-EG2机器人注册模块 - 直接导入本地的模块
try:
    # 使用本地的注册函数和机器人类
    from register_gen72_robot import GEN72EG2Robot, register_to_envs
    
    # 确保机器人已注册到环境中
    register_to_envs()
    
    # 获取机器人的UID
    robot_uid = GEN72EG2Robot.uid
    print(f"成功导入GEN72-EG2机器人，UID: {robot_uid}")
except ImportError as e:
    print(f"导入机器人注册模块失败: {e}")
    robot_uid = "gen72_eg2_robot"  # 假设已经通过其他方式注册

# 导入新的带相机的 Agent 类
try:
    # 确保从 register_gen72_robot 导入带相机版本
    from register_gen72_robot import GEN72EG2RobotWristCam, register_to_envs
    # 确保机器人已注册 (如果需要，调整 register_to_envs)
    # 假设 register_to_envs 会处理所有已注册的Agent或者装饰器会自动处理
    register_to_envs()
    robot_wristcam_uid = GEN72EG2RobotWristCam.uid
    print(f"成功导入带腕部相机的 GEN72-EG2 机器人，UID: {robot_wristcam_uid}")
except ImportError as e:
    print(f"导入带相机机器人失败: {e}")
    robot_wristcam_uid = "gen72_eg2_robot_wristcam" # 备用UID

@register_env("TwoRobotStackCube-v1", max_episode_steps=100)
class TwoRobotStackCubeEnv(BaseEnv):
    """
    双臂堆叠立方体环境 - 使用 GEN72-EG2 机器人
    
    任务描述: 两个机器人合作堆叠两个立方体。一个机器人需要拾取蓝色立方体放在目标区域，
    另一个机器人需要拾取绿色立方体并放置在蓝色立方体上方。
    """
    
    # 支持的机器人配置 - 使用新的带相机的 UID
    SUPPORTED_ROBOTS = [(robot_wristcam_uid, robot_wristcam_uid)]
    
    # 目标区域大小
    goal_radius = 0.06
    
    def __init__(
        self,
        *args,
        # 默认使用带相机的机器人
        robot_uids=(robot_wristcam_uid, robot_wristcam_uid),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = 0 # robot_init_qpos_noise # 设置为0以使用默认笔直姿态
        # 确保传入的 UID 是我们期望的带相机的 UID
        if robot_uids != (robot_wristcam_uid, robot_wristcam_uid):
            print(f"警告：环境默认使用带相机机器人 {robot_wristcam_uid}，但收到了 {robot_uids}。将强制使用默认值。")
            robot_uids = (robot_wristcam_uid, robot_wristcam_uid)
        print(f"环境初始化中，使用的 robot_uids: {robot_uids}") # 确认使用的UID
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sim_config(self):
        """默认的模拟配置"""
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )
    
    @property
    def _default_sensor_configs(self):
        """默认的传感器配置 - 修改为包含 Agent 的相机配置"""
        # 环境自身的相机
        base_pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        sensor_configs = [CameraConfig("base_camera", base_pose, 128, 128, np.pi / 2, 0.01, 100)]

        # 添加来自 Agent 的相机配置
        # 注意：这里假设 self.agent 已经在 __init__ 中被创建
        # 这可能需要调整，或者在第一次访问此属性时进行惰性加载
        if hasattr(self, 'agent') and isinstance(self.agent, MultiAgent):
            for agent_idx, agent_instance in enumerate(self.agent.agents):
                if hasattr(agent_instance, '_sensor_configs'):
                    try:
                        agent_configs = agent_instance._sensor_configs
                        if agent_configs:
                            # 确保 UID 是唯一的 (Agent类应该已经处理好了)
                            sensor_configs.extend(agent_configs)
                            print(f"已将 Agent {agent_idx} 的 {len(agent_configs)} 个传感器配置添加到环境配置中。")
                    except Exception as e:
                        print(f"警告: 获取 Agent {agent_idx} 的 _sensor_configs 时出错: {e}")
        else:
            print("警告: _default_sensor_configs 访问时 self.agent 尚未初始化或不是 MultiAgent 类型。")
            
        print(f"最终环境传感器配置数量: {len(sensor_configs)}")
        return sensor_configs
    
    @property
    def _default_human_render_camera_configs(self):
        """默认的人类渲染相机配置"""
        # 调整相机位置：更靠近正前方，稍微抬高，看这桌面中心
        # eye=[0.8, 0.0, 0.6] -> 在x轴正方向0.8米，高度0.6米
        # target=[0.0, 0.0, 0.0] -> 看向原点（桌面中心附近）
        pose = sapien_utils.look_at(eye=[0.9, -0.4, 0.8], target=[0.0, 0.0, 0.0])
        print("成功配置默认的人类渲染相机")
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    def _load_agent(self, options: dict):
        """加载机器人智能体"""
        # 为左右机器人设置初始位姿
        print(f"正在加载机器人，使用的UID: {self.robot_uids}")
        super()._load_agent(
            options, [sapien.Pose(p=[0, -0.25, 0]), sapien.Pose(p=[0, 0.25, 0])]
        )
        print("机器人加载完成。")
        
        # --- 恢复并改进: 在环境层面手动初始化Agent的传感器 ---
        print("在环境 _load_agent 中手动初始化 Agent 传感器...")
        from mani_skill.sensors.camera import Camera, CameraConfig
        
        if not hasattr(self.agent, 'agents') or not self.agent.agents:
            print("错误: self.agent 或 self.agent.agents 未初始化或为空。")
            return
            
        for agent_idx, agent in enumerate(self.agent.agents):
            agent_uid_name = agent.uid # 这是 Agent 实例的 UID (e.g., 'agent_0')
            print(f"处理 Agent {agent_idx} (实例UID: {agent_uid_name})...")
        
            # 确保 agent 有 sensors 字典
            if not hasattr(agent, 'sensors') or agent.sensors is None:
                agent.sensors = {}
                
            # 检查 agent 的类型是否有 _sensor_configs 属性
            # (应该访问 agent 实例自身的配置，而不是类属性)
            if not hasattr(agent, '_sensor_configs') or not agent._sensor_configs:
                 print(f"警告: Agent 实例 {agent_uid_name} 没有有效的 _sensor_configs 属性或配置为空，跳过。")
                 continue
                 
            try:
                sensor_configs_list = agent._sensor_configs
                print(f"Agent 实例 {agent_uid_name} 定义了 {len(sensor_configs_list)} 个传感器配置。")

                for config in sensor_configs_list:
                    # 使用 config.uid (例如 'hand_camera_agent_0') 作为字典键
                    sensor_key = config.uid 
                    if isinstance(config, CameraConfig):
                        try:
                            if sensor_key in agent.sensors:
                                print(f"信息: 传感器 '{sensor_key}' 已存在于 Agent {agent_uid_name} 的 sensors 字典中。")
                                continue
                                
                            print(f"尝试为 Agent {agent_uid_name} 创建并存储传感器: {sensor_key}")
                            # 注意: 这里的 articulation 应该是 agent.robot
                            if hasattr(agent, 'robot'):
                                camera_sensor = Camera(config, self.scene, articulation=agent.robot)
                                agent.sensors[sensor_key] = camera_sensor
                                print(f"成功为 Agent {agent_uid_name} 创建并添加传感器: {sensor_key}")
                            else:
                                print(f"错误: Agent {agent_uid_name} 缺少 'robot' 属性，无法创建相机。")
                        except Exception as e:
                            import traceback
                            print(f"错误: 为 Agent {agent_uid_name} 创建或存储传感器 '{sensor_key}' 失败: {e}")
                            print(traceback.format_exc())
                    else:
                        print(f"警告: 在 Agent {agent_uid_name} 的 _sensor_configs 中找到非 CameraConfig 类型: {type(config)}")
            except Exception as e:
                import traceback
                print(f"错误: 处理 Agent {agent_uid_name} 的 _sensor_configs 时出错: {e}")
                print(traceback.format_exc())
        # --- 传感器初始化结束 ---
        print("传感器初始化结束$$$$$$$$$$$$$$$")

    def _load_scene(self, options: dict):
        """加载场景"""
        # 设置立方体尺寸
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        
        # 使用TableSceneBuilder创建桌面场景
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        
        try:
            # 构建桌面场景
            self.table_scene.build()
            
            # --- 修改: 创建自定义尺寸和颜色的立方体 ---
            # 创建绿色立方体 A (4cm x 4cm x 2cm)
            builderA = self.scene.create_actor_builder()
            half_size_A = [0.02, 0.02, 0.01]
            builderA.add_box_collision(half_size=half_size_A)
            builderA.add_box_visual(half_size=half_size_A, material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])) # 绿色
            # --- 修改: 使用 initial_pose 设置初始位置 (x=0.85, y=-0.15) ---
            builderA.initial_pose = sapien.Pose(p=[0.55, -0.15, half_size_A[2]])
            # --- 修改结束 ---
            self.cubeA = builderA.build(name="cubeA")
            # --- 移除: 不再需要 set_pose ---
            # self.cubeA.set_pose(sapien.Pose(p=[0.85, -0.15, half_size_A[2]]))
            # --- 移除结束 ---

            # 创建橙色立方体 B (8cm x 4cm x 2cm)
            builderB = self.scene.create_actor_builder()
            half_size_B = [0.04, 0.02, 0.01]
            builderB.add_box_collision(half_size=half_size_B)
            builderB.add_box_visual(half_size=half_size_B, material=sapien.render.RenderMaterial(base_color=[1, 0.3, 0, 1])) # 橙色
            # --- 修改: 使用 initial_pose 设置初始位置 (x=0.85, y=0.0) ---
            builderB.initial_pose = sapien.Pose(p=[0.55, 0.0, half_size_B[2]])
            # --- 修改结束 ---
            self.cubeB = builderB.build(name="cubeB")
            # --- 移除: 不再需要 set_pose ---
            # self.cubeB.set_pose(sapien.Pose(p=[0.85, 0.0, half_size_B[2]]))
            # --- 移除结束 ---

            # --- 保留: 创建柠檬和香蕉 ---
            # 使用正确的加载方法和模型ID
            try:
                # 获取柠檬的 builder
                lemon_builder = actors.get_actor_builder(
                    self.scene,
                    id="ycb:014_lemon" # 正确的 YCB 模型 ID
                )
                # --- 修改: 调整初始位置 (x=0.85, y=0.25) ---
                lemon_builder.initial_pose = sapien.Pose(p=[0.5, -0.25, 0.025])
                # --- 修改结束 ---
                # 构建 actor
                self.lemon = lemon_builder.build(name="lemon")
                print("成功加载柠檬 (014_lemon)。")
            except Exception as e:
                print(f"加载柠檬 (014_lemon) 失败: {e}. 请确认模型是否存在于 ManiSkill YCB 资源中。")
                self.lemon = None # 标记为未加载

            try:
                # 获取香蕉的 builder
                banana_builder = actors.get_actor_builder(
                    self.scene,
                    id="ycb:011_banana" # 正确的 YCB 模型 ID
                )
                # --- 修改: 调整初始位置 (x=0.85, y=0.15) ---
                banana_builder.initial_pose = sapien.Pose(p=[0.5, 0.15, 0.025])
                # --- 修改结束 ---
                # 构建 actor
                self.banana = banana_builder.build(name="banana")
                print("成功加载香蕉 (011_banana)。")
            except Exception as e:
                print(f"加载香蕉 (011_banana) 失败: {e}. 请确认模型是否存在于 ManiSkill YCB 资源中。")
                self.banana = None # 标记为未加载
            # --- 添加结束 ---
        
        except Exception as e:
            print(f"场景加载错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        print("加载场景成功$$$$$$$$$$$$$$$")
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每个回合"""
        with torch.device(self.device):
            # 获取批次大小
            b = len(env_idx)
            
            try:
                # 初始化桌面场景 (这会重置桌子位置)
                self.table_scene.initialize(env_idx)

                # --- 添加: 在初始化后移动桌子 ---
                # 将桌子沿 X 轴正方向移动 0.7 米
                if hasattr(self.table_scene, 'table') and self.table_scene.table is not None:
                    table_actor = self.table_scene.table # 直接访问属性
                    # 获取 initialize 后的当前位姿 (此时是 [-0.12, 0, -0.91...])
                    current_pose = table_actor.pose 
                    
                    import numpy as np # 确保导入 numpy
                    current_p_tensor = current_pose.p
                    current_q_tensor = current_pose.q
                    
                    # 转换为 NumPy float32 数组
                    current_p_np = current_p_tensor.cpu().numpy().astype(np.float32).squeeze()
                    current_q_np = current_q_tensor.cpu().numpy().astype(np.float32).squeeze()
                    
                    translation_p = np.array([0.7, 0, 0], dtype=np.float32) 
                    
                    # 确保形状正确
                    if current_p_np.ndim > 1: current_p_np = current_p_np[0]
                    if current_q_np.ndim > 1: current_q_np = current_q_np[0]
                        
                    new_p_np = current_p_np + translation_p
                    
                    new_pose = sapien.Pose(p=new_p_np, q=current_q_np) 
                    
                    table_actor.set_pose(new_pose)
                    # 可以在这里再次打印确认，但为了简洁暂时省略
                    # print(f"在 initialize 后移动桌子到新位置: {new_pose.p}")
                else:
                    print("警告: _initialize_episode 中无法访问 self.table_scene.table。")
                # --- 添加结束 ---
            
                # --- 修改: 为新物体添加初始化逻辑 ---
                # (为简洁起见，这里暂时不随机化柠檬和香蕉的位置，
                #  它们将保持在 _load_scene 中设置的初始位置。
                #  如果需要随机化，需要在这里添加类似立方体的随机化代码)
                # 确保物体存在再进行操作
                if self.lemon is not None:
                    # 如果需要设置特定姿态 (例如，随机化)
                    # pose_lemon = ...
                    # self.lemon.set_pose(pose_lemon)
                    pass # 保持初始位置

                if self.banana is not None:
                    # 如果需要设置特定姿态
                    # pose_banana = ...
                    # self.banana.set_pose(pose_banana)
                    pass # 保持初始位置
                # --- 修改结束 ---
                
                # --- 移除: 立方体位置随机化 ---
                # # 设置蓝色立方体的位置 (现在是绿色)
                # cubeA_xyz = torch.zeros((b, 3), device=self.device)
                # cubeA_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.1 - 0.05
                # cubeA_xyz[:, 1] = -0.15 - torch.rand((b,), device=self.device) * 0.1 + 0.05
                # cubeA_xyz[:, 2] = 0.02
                # 
                # # 生成随机旋转
                # qs = random_quaternions(
                #     b,
                #     lock_x=True,
                #     lock_y=True,
                #     lock_z=False,
                #     device=self.device
                # )
                # 
                # # 设置蓝色立方体的姿态
                # self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_xyz, q=qs))
                # 
                # # 设置绿色立方体的位置 (现在是橙色)
                # cubeB_xyz = torch.zeros((b, 3), device=self.device)
                # cubeB_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.1 - 0.05
                # cubeB_xyz[:, 1] = 0.15 + torch.rand((b,), device=self.device) * 0.1 - 0.05
                # cubeB_xyz[:, 2] = 0.02
                # 
                # # 生成随机旋转
                # qs = random_quaternions(
                #     b,
                #     lock_x=True,
                #     lock_y=True,
                #     lock_z=False,
                #     device=self.device
                # )
                # 
                # # 设置绿色立方体的姿态
                # self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_xyz, q=qs))
                # --- 移除结束 ---
                
                # --- 移除: 目标区域位置设置 ---
                # target_region_xyz = torch.zeros((b, 3), device=self.device)
                # target_region_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.1 - 0.05
                # target_region_xyz[:, 1] = 0.0  # 居中放置
                # target_region_xyz[..., 2] = 1e-3  # 略高于桌面
                # 
                # # 设置目标区域的姿态
                # self.goal_region.set_pose(
                #     Pose.create_from_pq(
                #         p=target_region_xyz,
                #         q=euler2quat(0, np.pi / 2, 0),
                #     )
                # )
                # --- 移除结束 ---
            except Exception as e:
                print(f"初始化回合时出错: {e}")
                import traceback
                traceback.print_exc()
                raise
        print("初始化每个回合###############")
    
    @property
    def left_agent(self):
        """获取左侧机器人"""
        return self.agent.agents[0]
    
    @property
    def right_agent(self):
        """获取右侧机器人"""
        return self.agent.agents[1]
    
    def evaluate(self):
        """评估当前状态"""
        # 获取立方体位置
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        
        # 计算立方体之间的偏移
        offset = pos_A - pos_B
        
        # 检查XY平面上立方体是否对齐
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        
        # 检查Z轴上立方体是否堆叠
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        
        # 判断立方体A是否在立方体B上方
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        
        # 判断立方体是否被抓取
        is_cubeA_grasped = self.left_agent.is_grasping(self.cubeA)
        is_cubeB_grasped = self.right_agent.is_grasping(self.cubeB)
        
        # 判断任务是否成功完成 (修改：移除 cubeB_placed)
        success = (
            is_cubeA_on_cubeB * (~is_cubeA_grasped) * (~is_cubeB_grasped)
        )
        print("评估当前状态！！！！！！！！！！！")
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeB_grasped": is_cubeB_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "success": success.bool(),
        }
        
    def _get_obs_extra(self, info: dict):
        """获取额外的观察信息"""
        # BaseEnv 会自动收集传感器数据 (包括相机)
        # 这个方法只用于添加非传感器、非 qpos/qvel 的状态信息
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                left_arm_tcp_to_cubeA_pos=self.cubeA.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cubeB_pos=self.cubeB.pose.p - self.right_agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        print("获取额外的观察信息$$$$$$$$")
        return obs
        
    def compute_dense_reward(self, obs: any, action: torch.Tensor, info: dict):
        """计算密集奖励"""
        # 阶段1: 接近并抓取
        # 计算左机械臂到立方体A的距离
        cubeA_to_left_arm_tcp_dist = torch.linalg.norm(
            self.left_agent.tcp.pose.p - self.cubeA.pose.p, axis=1
        )
        
        # 计算右机械臂推动位置的姿态
        right_arm_push_pose = Pose.create_from_pq(
            p=self.cubeB.pose.p + torch.tensor([0, self.cube_half_size[0] + 0.005, 0], device=self.device)
        )
        
        # 计算右机械臂到推动位置的距离
        right_arm_to_push_pose_dist = torch.linalg.norm(
            right_arm_push_pose.p - self.right_agent.tcp.pose.p, axis=1
        )
        
        # 计算接近奖励
        reach_reward = (
            1 - torch.tanh(5 * cubeA_to_left_arm_tcp_dist)
            + 1 - torch.tanh(5 * right_arm_to_push_pose_dist)
        ) / 2
        
        # 立方体位置
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        
        # 初始奖励
        reward = (reach_reward + info["is_cubeA_grasped"]) / 2
        
        # 阶段1通过条件
        place_stage_reached = info["is_cubeA_grasped"]
        
        # --- 修改: 调整后续阶段的触发条件和奖励值 (需要重新设计) ---
        # 临时方案：假设阶段 2 现在是 cubeB 在某个大致位置 (这里简化，直接进入下一阶段判断)
        # 触发条件直接使用 place_stage_reached (即 cubeA 被抓住)
        cubeB_placed_and_cubeA_grasped = place_stage_reached # 临时替代
        
        # 阶段3: 放置顶部立方体，同时右机械臂移开给左机械臂留出空间
        # (这部分逻辑可能仍然相关，但基准奖励值需要调整)
        # 计算立方体A的目标位置 (堆叠在B上)
        goal_xyz = torch.hstack(
            [cubeB_pos[:, :2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]] # 使用 self.cube_half_size[2] (cubeB的半高)
        )
        
        # 计算立方体A到目标位置的距离
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        
        # 放置奖励 (堆叠)
        stack_place_reward = 1 - torch.tanh(5 * cubeA_to_goal_dist)
        
        # 计算右机械臂离开奖励
        right_arm_leave_reward = 1 - torch.tanh(
            5 * (self.right_agent.tcp.pose.p[:, 1] - 0.2).abs() # 让右臂向 y=0.2 方向移动
        )
        
        # 阶段3奖励
        stage_3_reward = stack_place_reward * 2 + right_arm_leave_reward
        
        # 更新奖励 (基于临时触发条件)
        # 调整基准奖励值 (原为 4，现在从 2 开始计算)
        reward[cubeB_placed_and_cubeA_grasped] = (
            2 + stage_3_reward[cubeB_placed_and_cubeA_grasped]
        ) 
        
        # 阶段3通过条件 (现在是 is_cubeA_on_cubeB)
        cubes_placed = info["is_cubeA_on_cubeB"] # 使用评估结果
        
        # 阶段4: 使两个机械臂停止抓取 (逻辑不变，调整基准奖励值)
        # 计算夹爪宽度
        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        
        # 左机械臂松开奖励
        ungrasp_reward_left = (
            torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        # 如果一开始就没抓，奖励为1
        ungrasp_reward_left[~info["is_cubeA_grasped"]] = 1.0 
        
        # 右机械臂松开奖励
        ungrasp_reward_right = (
            torch.sum(self.right_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        # 如果一开始就没抓，奖励为1
        ungrasp_reward_right[~info["is_cubeB_grasped"]] = 1.0
        
        # 更新通过阶段3的奖励 (调整基准奖励值，原为 8，现在从 4 开始)
        reward[cubes_placed] = (
            4 + (ungrasp_reward_left + ungrasp_reward_right)[cubes_placed] / 2
        )
        
        # 成功完成奖励 (调整基准奖励值，原为 10，现在为 6)
        reward[info["success"]] = 6
        print("计算密集奖励$$$$$$$$")
        return reward
    
    def compute_normalized_dense_reward(self, obs: any, action: torch.Tensor, info: dict):
        """计算归一化密集奖励"""
        # 调整归一化因子 (原为 10，现在为 6)
        print("计算归一化密集奖励$$$$$$$$")
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6 