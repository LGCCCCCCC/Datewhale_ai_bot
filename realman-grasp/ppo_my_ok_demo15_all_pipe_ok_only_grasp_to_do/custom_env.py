#!/usr/bin/env python3
"""
自定义ManiSkill环境，针对GEN72-EG2机器人优化 (支持多臂)
"""

import numpy as np
import sapien
import torch
from mani_skill.utils import common
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from copy import deepcopy
from mani_skill.envs.sapien_env import BaseEnv
from typing import Optional, Tuple, Any, Dict, List

# Helper function to create poses (optional, but keeps code cleaner)
def create_sapien_pose(xyz):
    return sapien.Pose(p=xyz)

@register_env("CustomPushCube-v1")
class CustomPushCubeEnv(PushCubeEnv):
    """针对GEN72-EG2机器人优化的PushCube环境 (支持多臂)"""
    
    # 支持的机器人配置
    SUPPORTED_ROBOTS = ["gen72_eg2_robot"]  # 单臂
    # 注意：在支持的机器人列表中添加元组，表示支持多臂配置
    SUPPORTED_ROBOTS.append(("gen72_eg2_robot", "gen72_eg2_robot"))  # 双臂

    def _load_agent(self, options=None):
        """加载机械臂，支持单臂或双臂配置"""
        # 判断是单臂还是双臂模式
        num_robots = len(self.robot_uids)
        
        # 准备初始位姿
        initial_agent_poses = []
        if num_robots == 1:
            # 单机器人，使用默认或自定义位姿
            initial_agent_poses.append(create_sapien_pose([-0.615, 0, 0]))
        elif num_robots == 2:
            print("为两个机器人准备初始位姿...")
            # 两个机器人的位姿，左右布局
            initial_agent_poses.append(create_sapien_pose([-0.5, -0.4, 0]))  # 左侧机器人
            initial_agent_poses.append(create_sapien_pose([-0.5, 0.4, 0]))   # 右侧机器人
        else:
            print(f"警告: 当前只为1或2个机器人配置了初始位姿，收到了 {num_robots} 个。使用默认位姿。")
            for _ in range(num_robots):
                initial_agent_poses.append(sapien.Pose())

        print(f"传递 {len(initial_agent_poses)} 个初始位姿给基类加载: {initial_agent_poses}")
        # 调用基类 (BaseEnv) 的 _load_agent 方法
        super(PushCubeEnv, self)._load_agent(options=options, initial_agent_poses=initial_agent_poses)

    def _get_obs_extra(self, info: Dict) -> Dict:
        """计算多智能体的附加观测信息"""
        # 初始化观测字典
        extra = {}
        
        # 判断是单臂还是双臂模式
        if hasattr(self, 'agents') and len(self.agents) > 1:
            # 多机器人模式，类似于 TwoRobotStackCube
            left_agent = self.agents[0]
            right_agent = self.agents[1]
            
            # 为每个机器人收集 TCP 位姿
            extra["left_arm_tcp"] = left_agent.tcp.pose.raw_pose
            extra["right_arm_tcp"] = right_agent.tcp.pose.raw_pose
            
            # 如果需要状态信息
            if self._obs_mode in ["state", "state_dict"]:
                if hasattr(self, "obj"):
                    extra["obj_pose"] = self.obj.pose.raw_pose
                    # 计算物体与机械臂末端的相对位置
                    extra["left_arm_tcp_to_obj_pos"] = self.obj.pose.p - left_agent.tcp.pose.p
                    extra["right_arm_tcp_to_obj_pos"] = self.obj.pose.p - right_agent.tcp.pose.p
                
                if hasattr(self, "goal_pos"):
                    extra["goal_pos"] = self.goal_pos
                    if hasattr(self, "obj"):
                        # 物体与目标的相对位置
                        extra["obj_to_goal_pos"] = self.goal_pos - self.obj.pose.p
        else:
            # 单机器人后备模式
            try:
                # 尝试直接访问 self.agent
                extra["tcp_pose"] = self.agent.tcp.pose.raw_pose
                
                if self._obs_mode in ["state", "state_dict"]:
                    if hasattr(self, "obj"):
                        extra["obj_pose"] = self.obj.pose.raw_pose
                        extra["tcp_to_obj_pos"] = self.obj.pose.p - self.agent.tcp.pose.p
                    
                    if hasattr(self, "goal_pos"):
                        extra["goal_pos"] = self.goal_pos
                        if hasattr(self, "obj"):
                            extra["obj_to_goal_pos"] = self.goal_pos - self.obj.pose.p
            except AttributeError:
                # 如果 self.agent 不存在或不具有 tcp 属性
                print("警告: 单机器人模式下无法获取 self.agent.tcp")
            
                # 如果有 self.agents，退回到使用第一个 agent
                if hasattr(self, 'agents') and len(self.agents) > 0:
                    agent = self.agents[0]
                    extra["tcp_pose"] = agent.tcp.pose.raw_pose
                    
                    if self._obs_mode in ["state", "state_dict"]:
                        if hasattr(self, "obj"):
                            extra["obj_pose"] = self.obj.pose.raw_pose
                            extra["tcp_to_obj_pos"] = self.obj.pose.p - agent.tcp.pose.p
                        
                        if hasattr(self, "goal_pos"):
                            extra["goal_pos"] = self.goal_pos
                            if hasattr(self, "obj"):
                                extra["obj_to_goal_pos"] = self.goal_pos - self.obj.pose.p
        
        return extra

    @property
    def left_agent(self):
        """获取左侧机器人（如果有）"""
        if hasattr(self, 'agents') and len(self.agents) > 0:
            return self.agents[0]
        return None
    
    @property
    def right_agent(self):
        """获取右侧机器人（如果有）"""
        if hasattr(self, 'agents') and len(self.agents) > 1:
            return self.agents[1]
        return None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """重写初始化函数，仅优化物体/目标位置"""
        super()._initialize_episode(env_idx, options)
        
        b = len(env_idx)
        if hasattr(self, "obj"):
            new_cube_pos = torch.zeros((b, 3), device=self.device)
            new_cube_pos[:, 0] = -0.239
            new_cube_pos[:, 1] = 0.030
            new_cube_pos[:, 2] = 0.025
            new_cube_pose = Pose.create_from_pq(p=new_cube_pos, q=[1, 0, 0, 0])
            self.obj.set_pose(new_cube_pose)
            print("已调整方块初始位置 (_initialize_episode)。")
            
        if hasattr(self, "goal_region"):
            new_target_pos = torch.zeros((b, 3), device=self.device)
            new_target_pos[:, 0] = -0.100
            new_target_pos[:, 1] = 0.000
            new_target_pos[:, 2] = 0.001
            target_quat = euler2quat(0, np.pi / 2, 0)
            new_target_pose = Pose.create_from_pq(p=new_target_pos, q=target_quat)
            self.goal_region.set_pose(new_target_pose)
            print("已调整目标初始位置 (_initialize_episode)。")


# --- 对 CustomPickCubeEnv 进行类似的修改 ---

@register_env("CustomPickCube-v1")
class CustomPickCubeEnv(PickCubeEnv):
    """针对GEN72-EG2机器人优化的PickCube环境 (支持多臂)"""
    
    # 支持的机器人配置
    SUPPORTED_ROBOTS = ["gen72_eg2_robot"]  # 单臂
    # 注意：在支持的机器人列表中添加元组，表示支持多臂配置
    SUPPORTED_ROBOTS.append(("gen72_eg2_robot", "gen72_eg2_robot"))  # 双臂

    def _load_agent(self, options=None):
        """加载机械臂，支持单臂或双臂配置"""
        # 判断是单臂还是双臂模式
        num_robots = len(self.robot_uids)
        
        # 准备初始位姿
        initial_agent_poses = []
        if num_robots == 1:
            # 单机器人，使用默认或自定义位姿
            initial_agent_poses.append(create_sapien_pose([-0.58, 0, 0]))
        elif num_robots == 2:
            print("为两个机器人准备初始位姿...")
            # 两个机器人的位姿，左右布局
            initial_agent_poses.append(create_sapien_pose([-0.5, -0.4, 0]))  # 左侧机器人
            initial_agent_poses.append(create_sapien_pose([-0.5, 0.4, 0]))   # 右侧机器人
        else:
            print(f"警告: 当前只为1或2个机器人配置了初始位姿，收到了 {num_robots} 个。使用默认位姿。")
            for _ in range(num_robots):
                initial_agent_poses.append(sapien.Pose())

        print(f"传递 {len(initial_agent_poses)} 个初始位姿给基类加载: {initial_agent_poses}")
        # 调用基类 (BaseEnv) 的 _load_agent 方法
        super(PickCubeEnv, self)._load_agent(options=options, initial_agent_poses=initial_agent_poses)

    def _get_obs_extra(self, info: Dict) -> Dict:
        """计算多智能体的附加观测信息"""
        # 初始化观测字典
        extra = {}
        
        # 判断是单臂还是双臂模式
        if hasattr(self, 'agents') and len(self.agents) > 1:
            # 多机器人模式，类似于 TwoRobotStackCube
            left_agent = self.agents[0]
            right_agent = self.agents[1]
            
            # 为每个机器人收集 TCP 位姿
            extra["left_arm_tcp"] = left_agent.tcp.pose.raw_pose
            extra["right_arm_tcp"] = right_agent.tcp.pose.raw_pose
            
            # 抓取状态
            if hasattr(self, "cube"):
                extra["left_arm_is_grasping"] = left_agent.check_grasp(self.cube)
                extra["right_arm_is_grasping"] = right_agent.check_grasp(self.cube)
            
            # 如果需要状态信息
            if self._obs_mode in ["state", "state_dict"]:
                if hasattr(self, "cube"):
                    extra["cube_pose"] = self.cube.pose.raw_pose
                    # 计算物体与机械臂末端的相对位置
                    extra["left_arm_tcp_to_cube_pos"] = self.cube.pose.p - left_agent.tcp.pose.p
                    extra["right_arm_tcp_to_cube_pos"] = self.cube.pose.p - right_agent.tcp.pose.p
                
                if hasattr(self, "goal_site") and hasattr(self, "cube"):
                    # 物体与目标的相对位置
                    extra["cube_to_goal_pos"] = self.goal_site.pose.p - self.cube.pose.p
        else:
            # 单机器人后备模式
            try:
                # 尝试直接访问 self.agent
                extra["tcp_pose"] = self.agent.tcp.pose.raw_pose
                
                if hasattr(self, "cube"):
                    extra["is_grasped"] = self.agent.check_grasp(self.cube)
                
                if self._obs_mode in ["state", "state_dict"]:
                    if hasattr(self, "cube"):
                        extra["cube_pose"] = self.cube.pose.raw_pose
                        extra["tcp_to_cube_pos"] = self.cube.pose.p - self.agent.tcp.pose.p
                    
                    if hasattr(self, "goal_site") and hasattr(self, "cube"):
                        extra["cube_to_goal_pos"] = self.goal_site.pose.p - self.cube.pose.p
            except AttributeError:
                # 如果 self.agent 不存在或不具有 tcp 属性
                print("警告: 单机器人模式下无法获取 self.agent.tcp")
                
                # 如果有 self.agents，退回到使用第一个 agent
                if hasattr(self, 'agents') and len(self.agents) > 0:
                    agent = self.agents[0]
                    extra["tcp_pose"] = agent.tcp.pose.raw_pose
                    
                    if hasattr(self, "cube"):
                        extra["is_grasped"] = agent.check_grasp(self.cube)
                    
                    if self._obs_mode in ["state", "state_dict"]:
                        if hasattr(self, "cube"):
                            extra["cube_pose"] = self.cube.pose.raw_pose
                            extra["tcp_to_cube_pos"] = self.cube.pose.p - agent.tcp.pose.p
                        
                        if hasattr(self, "goal_site") and hasattr(self, "cube"):
                            extra["cube_to_goal_pos"] = self.goal_site.pose.p - self.cube.pose.p
        
        return extra

    @property
    def left_agent(self):
        """获取左侧机器人（如果有）"""
        if hasattr(self, 'agents') and len(self.agents) > 0:
            return self.agents[0]
        return None
    
    @property
    def right_agent(self):
        """获取右侧机器人（如果有）"""
        if hasattr(self, 'agents') and len(self.agents) > 1:
            return self.agents[1]
        return None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """重写初始化函数，仅优化物体/目标位置"""
        super()._initialize_episode(env_idx, options)

            b = len(env_idx)
        if hasattr(self, "cube"):
            new_cube_pos = torch.zeros((b, 3), device=self.device)
            new_cube_pos[:, 0] = -0.239
            new_cube_pos[:, 1] = 0.030
            new_cube_pos[:, 2] = 0.025
            new_cube_pose = Pose.create_from_pq(p=new_cube_pos, q=[1, 0, 0, 0])
            self.cube.set_pose(new_cube_pose)
            print("已调整方块初始位置 (_initialize_episode)。")
            
        if hasattr(self, "target_site"):
                new_target_pos = torch.zeros((b, 3), device=self.device)
            new_target_pos[:, 0] = -0.100
            new_target_pos[:, 1] = 0.000
            new_target_pos[:, 2] = 0.1 # PickCube 目标位置在空中
                target_quat = euler2quat(0, np.pi / 2, 0)
                new_target_pose = Pose.create_from_pq(p=new_target_pos, q=target_quat)
            self.target_site.set_pose(new_target_pose)
            print("已调整目标初始位置 (_initialize_episode)。")


# =====================================================
# 新增专用双机械臂环境 - 类似于 TwoRobotStackCube
# =====================================================

@register_env("TwoRobotPushCube-v1")
class TwoRobotPushCubeEnv(PushCubeEnv):
    """专为双机械臂设计的推方块环境"""
    
    # 仅支持双臂配置
    SUPPORTED_ROBOTS = [("gen72_eg2_robot", "gen72_eg2_robot")]
    
    def __init__(self, *args, **kwargs):
        """初始化环境，确保使用双机器人配置"""
        super().__init__(*args, **kwargs)
        print("初始化双机械臂推方块环境...")
        
        # 确保是双机器人模式
        if not isinstance(self.robot_uids, tuple) or len(self.robot_uids) != 2:
            raise ValueError("TwoRobotPushCubeEnv 仅支持双机器人配置！请使用 robot_uids=('gen72_eg2_robot', 'gen72_eg2_robot')")
    
    def _load_agent(self, options=None):
        """加载双机械臂"""
        print("加载双机械臂...")
        
        # 准备双臂初始位姿 - 左右布局
        initial_agent_poses = [
            create_sapien_pose([-0.5, -0.4, 0]),  # 左侧机器人
            create_sapien_pose([-0.5, 0.4, 0])    # 右侧机器人
        ]
        
        print(f"传递双臂初始位姿给基类加载: {initial_agent_poses}")
        super(PushCubeEnv, self)._load_agent(options=options, initial_agent_poses=initial_agent_poses)
        
        # 环境加载后检查是否生成了 agents 列表
        if not hasattr(self, 'agents') or len(self.agents) != 2:
            print(f"警告: 环境未正确生成双机械臂! agents 列表: {getattr(self, 'agents', '不存在')}")
        else:
            print(f"成功加载双机械臂，agents 数量: {len(self.agents)}")
    
    def _get_obs_extra(self, info: Dict) -> Dict:
        """计算双机械臂的观测信息"""
        # 初始化观测字典
        extra = {}
        
        try:
            # 确保 agents 列表存在且包含两个机器人
            if hasattr(self, 'agents') and len(self.agents) == 2:
                left_agent = self.agents[0]
                right_agent = self.agents[1]
                
                # 收集基础信息
                extra["left_arm_tcp"] = left_agent.tcp.pose.raw_pose
                extra["right_arm_tcp"] = right_agent.tcp.pose.raw_pose
                
                # 如果需要状态信息
                if self._obs_mode in ["state", "state_dict"]:
                    if hasattr(self, "obj"):
                        extra["obj_pose"] = self.obj.pose.raw_pose
                        extra["left_arm_tcp_to_obj_pos"] = self.obj.pose.p - left_agent.tcp.pose.p
                        extra["right_arm_tcp_to_obj_pos"] = self.obj.pose.p - right_agent.tcp.pose.p
                    
                    if hasattr(self, "goal_pos"):
                        extra["goal_pos"] = self.goal_pos
                        if hasattr(self, "obj"):
                            extra["obj_to_goal_pos"] = self.goal_pos - self.obj.pose.p
            else:
                # 应急措施
                print(f"警告: 无法找到双机械臂! agents 列表: {getattr(self, 'agents', '不存在')}")
                if hasattr(self, "agent"):
                    # 单臂后备方案
                    extra["tcp_pose"] = self.agent.tcp.pose.raw_pose
                    
                    if self._obs_mode in ["state", "state_dict"]:
                        if hasattr(self, "obj"):
                            extra["obj_pose"] = self.obj.pose.raw_pose
                            extra["tcp_to_obj_pos"] = self.obj.pose.p - self.agent.tcp.pose.p
                        
                        if hasattr(self, "goal_pos"):
                            extra["goal_pos"] = self.goal_pos
                            if hasattr(self, "obj"):
                                extra["obj_to_goal_pos"] = self.goal_pos - self.obj.pose.p
        except Exception as e:
            print(f"获取观测时出错: {e}")
            # 提供最小可行的观测信息
            extra = {"error": True}
        
        return extra
    
    @property
    def left_agent(self):
        """获取左侧机器人"""
        if hasattr(self, 'agents') and len(self.agents) > 0:
            return self.agents[0]
        raise AttributeError("左侧机器人不存在")
    
    @property
    def right_agent(self):
        """获取右侧机器人"""
        if hasattr(self, 'agents') and len(self.agents) > 1:
            return self.agents[1]
        raise AttributeError("右侧机器人不存在")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化场景，设置物体和目标位置"""
        super()._initialize_episode(env_idx, options)
        
        # 设置物体位置
        b = len(env_idx)
        if hasattr(self, "obj"):
            new_cube_pos = torch.zeros((b, 3), device=self.device)
            new_cube_pos[:, 0] = -0.2  # 在双臂之间
            new_cube_pos[:, 1] = 0.0   # 居中位置
            new_cube_pos[:, 2] = 0.025
            new_cube_pose = Pose.create_from_pq(p=new_cube_pos, q=[1, 0, 0, 0])
            self.obj.set_pose(new_cube_pose)
            print("已调整方块初始位置 (双臂环境)。")
            
        # 设置目标位置
        if hasattr(self, "goal_region"):
            new_target_pos = torch.zeros((b, 3), device=self.device)
            new_target_pos[:, 0] = -0.1
            new_target_pos[:, 1] = 0.0
            new_target_pos[:, 2] = 0.001
            target_quat = euler2quat(0, np.pi / 2, 0)
            new_target_pose = Pose.create_from_pq(p=new_target_pos, q=target_quat)
            self.goal_region.set_pose(new_target_pose)
            print("已调整目标初始位置 (双臂环境)。")


@register_env("TwoRobotPickCube-v1")
class TwoRobotPickCubeEnv(PickCubeEnv):
    """专为双机械臂设计的抓取方块环境"""
    
    # 仅支持双臂配置
    SUPPORTED_ROBOTS = [("gen72_eg2_robot", "gen72_eg2_robot")]
    
    def __init__(self, *args, **kwargs):
        """初始化环境，确保使用双机器人配置"""
        super().__init__(*args, **kwargs)
        print("初始化双机械臂抓取方块环境...")
        
        # 确保是双机器人模式
        if not isinstance(self.robot_uids, tuple) or len(self.robot_uids) != 2:
            raise ValueError("TwoRobotPickCubeEnv 仅支持双机器人配置！请使用 robot_uids=('gen72_eg2_robot', 'gen72_eg2_robot')")
    
    def _load_agent(self, options=None):
        """加载双机械臂"""
        print("加载双机械臂...")
        
        # 准备双臂初始位姿 - 左右布局
        initial_agent_poses = [
            create_sapien_pose([-0.5, -0.4, 0]),  # 左侧机器人
            create_sapien_pose([-0.5, 0.4, 0])    # 右侧机器人
        ]
        
        print(f"传递双臂初始位姿给基类加载: {initial_agent_poses}")
        super(PickCubeEnv, self)._load_agent(options=options, initial_agent_poses=initial_agent_poses)
        
        # 环境加载后检查是否生成了 agents 列表
        if not hasattr(self, 'agents') or len(self.agents) != 2:
            print(f"警告: 环境未正确生成双机械臂! agents 列表: {getattr(self, 'agents', '不存在')}")
        else:
            print(f"成功加载双机械臂，agents 数量: {len(self.agents)}")
    
    def _get_obs_extra(self, info: Dict) -> Dict:
        """计算双机械臂的观测信息"""
        # 初始化观测字典
        extra = {}
        
        try:
            # 确保 agents 列表存在且包含两个机器人
            if hasattr(self, 'agents') and len(self.agents) == 2:
                left_agent = self.agents[0]
                right_agent = self.agents[1]
                
                # 收集基础信息
                extra["left_arm_tcp"] = left_agent.tcp.pose.raw_pose
                extra["right_arm_tcp"] = right_agent.tcp.pose.raw_pose
                
                # 抓取状态
                if hasattr(self, "cube"):
                    extra["left_arm_is_grasping"] = left_agent.check_grasp(self.cube)
                    extra["right_arm_is_grasping"] = right_agent.check_grasp(self.cube)
                
                # 如果需要状态信息
                if self._obs_mode in ["state", "state_dict"]:
                    if hasattr(self, "cube"):
                        extra["cube_pose"] = self.cube.pose.raw_pose
                        extra["left_arm_tcp_to_cube_pos"] = self.cube.pose.p - left_agent.tcp.pose.p
                        extra["right_arm_tcp_to_cube_pos"] = self.cube.pose.p - right_agent.tcp.pose.p
                    
                    if hasattr(self, "goal_site") and hasattr(self, "cube"):
                        extra["cube_to_goal_pos"] = self.goal_site.pose.p - self.cube.pose.p
            else:
                # 应急措施
                print(f"警告: 无法找到双机械臂! agents 列表: {getattr(self, 'agents', '不存在')}")
                if hasattr(self, "agent"):
                    # 单臂后备方案
                    extra["tcp_pose"] = self.agent.tcp.pose.raw_pose
                    
                    if hasattr(self, "cube"):
                        extra["is_grasped"] = self.agent.check_grasp(self.cube)
                    
                    if self._obs_mode in ["state", "state_dict"]:
                        if hasattr(self, "cube"):
                            extra["cube_pose"] = self.cube.pose.raw_pose
                            extra["tcp_to_cube_pos"] = self.cube.pose.p - self.agent.tcp.pose.p
                        
                        if hasattr(self, "goal_site") and hasattr(self, "cube"):
                            extra["cube_to_goal_pos"] = self.goal_site.pose.p - self.cube.pose.p
        except Exception as e:
            print(f"获取观测时出错: {e}")
            # 提供最小可行的观测信息
            extra = {"error": True}
        
        return extra
    
    @property
    def left_agent(self):
        """获取左侧机器人"""
        if hasattr(self, 'agents') and len(self.agents) > 0:
            return self.agents[0]
        raise AttributeError("左侧机器人不存在")
    
    @property
    def right_agent(self):
        """获取右侧机器人"""
        if hasattr(self, 'agents') and len(self.agents) > 1:
            return self.agents[1]
        raise AttributeError("右侧机器人不存在")
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化场景，设置物体和目标位置"""
        super()._initialize_episode(env_idx, options)
        
        # 设置物体位置
        b = len(env_idx)
        if hasattr(self, "cube"):
            new_cube_pos = torch.zeros((b, 3), device=self.device)
            new_cube_pos[:, 0] = -0.2  # 在双臂之间
            new_cube_pos[:, 1] = 0.0   # 居中位置
            new_cube_pos[:, 2] = 0.025
            new_cube_pose = Pose.create_from_pq(p=new_cube_pos, q=[1, 0, 0, 0])
            self.cube.set_pose(new_cube_pose)
            print("已调整方块初始位置 (双臂环境)。")
            
        # 设置目标位置
            if hasattr(self, "target_site"):
                new_target_pos = torch.zeros((b, 3), device=self.device)
            new_target_pos[:, 0] = -0.1
            new_target_pos[:, 1] = 0.0
            new_target_pos[:, 2] = 0.1  # PickCube 目标位置在空中
                target_quat = euler2quat(0, np.pi / 2, 0)
                new_target_pose = Pose.create_from_pq(p=new_target_pos, q=target_quat)
                self.target_site.set_pose(new_target_pose)
            print("已调整目标初始位置 (双臂环境)。") 