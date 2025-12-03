#!/usr/bin/env python3
"""
注册GEN72-EG2机器人到ManiSkill环境中

此脚本注册自定义的GEN72-EG2机器人，配置其物理属性和控制参数，
使其可以在ManiSkill环境中使用，特别是与稳定版PPO算法一起使用。
"""
import os
import sys
import numpy as np
# torch将在实际训练时导入，注册阶段可选
try:
    import torch
except ImportError:
    print("警告: torch未安装，仅在使用PPO训练时需要")
import sapien
from copy import deepcopy
from pathlib import Path
import mani_skill
from pathlib import Path
import re

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
    processed_path = original_urdf_path.parent / (original_urdf_path.stem + ".processed.urdf")

    # 简单缓存：如果已存在处理过的文件，并且比源文件新，则直接使用
    if processed_path.exists() and processed_path.stat().st_mtime > original_urdf_path.stat().st_mtime:
        return str(processed_path)

    print(f"--- Pre-processing URDF for SAPIEN: {original_urdf_path.name} ---")
    with open(original_urdf_path, "r") as f:
        content = f.read()

    # 关键替换步骤
    processed_content = content.replace(
        "package://mani_skill/", f"{str(_mani_skill_root_dir)}/"
    )

    with open(processed_path, "w") as f:
        f.write(processed_content)

    print(f"Saved processed URDF with absolute paths to: {processed_path.name}")
    return str(processed_path)

# 3. 定义原始URDF文件的路径并生成处理后的路径
_original_urdf_path = _project_root / "urdf_01/GEN72-EG2.urdf"
_processed_urdf_path = get_processed_urdf_path(_original_urdf_path)


# 确保可以导入ManiSkill模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig

# 修改这个路径为您的URDF文件实际路径
# URDF_PATH = '../urdf_01/GEN72-EG2.urdf' # 移除旧的URDF_PATH

# 环境第一次注册时可能会出现缺少"4C2_baselink"等链接的错误，这是因为URDF文件中的一些链接可能无法被物理引擎正确加载
# 如果遇到此类错误，可能需要修改URDF文件或检查URDF中的链接/关节定义是否有问题

@register_agent()
class GEN72EG2Robot(BaseAgent):
    """
    GEN72-EG2 机器人类，集成了7自由度机械臂和EG2夹爪
    """
    uid = "gen72_eg2_robot"
    urdf_path = _processed_urdf_path # 使用处理后的URDF路径
    
    # 设置摩擦力以便抓取物体
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            # 确保这些链接名称与URDF中的实际名称匹配
            "Link7": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "4C2_Link2": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            "4C2_Link3": dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        },
    )
    
    # 定义关节名称 - 从URDF文件中获取的实际关节名称
    arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    # 夹爪关节名称，从URDF文件 - 现在只包含驱动关节
    gripper_joint_names = ['4C2_Joint1']
    
    # 末端效应器链接名称
    ee_link_name = "Link7"
    tcp_link_name = "Link7"  # 定义TCP链接名称，通常与末端效应器相同
    
    # 控制参数 - 经过调整以确保更稳定的控制
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100
    
    # 定义初始姿态的关键帧 - 针对推方块任务优化的姿势
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                0, -0.1, 0, -1.5, 0, 1.8, 0.8,  # 机械臂 - 更自然的姿势，手臂稍微下垂，适合推方块
                0.82  # 夹爪驱动关节 (4C2_Joint1) 打开状态 (修改从闭合改为打开)
            ]),
            # 旋转180度并前移基座，使机械臂能轻松接触到桌面物体
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),  # 绕z轴旋转180度并前移0.2米
        ),
        push_ready=Keyframe(
            qpos=np.array([
                0, 0.2, 0, -1.2, 0, 1.6, 0.0,  # 机械臂 - 推动准备姿势
                0.82  # 夹爪驱动关节 (4C2_Joint1) 张开
            ]),
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),
        ),
        grasp_ready=Keyframe(
            qpos=np.array([
                0, 0.1, 0, -1.0, 0, 1.2, 0.0,  # 机械臂 - 抓取准备姿势
                0.82  # 夹爪驱动关节 (4C2_Joint1) 张开
            ]),
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),
        ),
        grasp_close=Keyframe(
            qpos=np.array([
                0, 0.1, 0, -1.0, 0, 1.2, 0.0,  # 机械臂 - 抓取准备姿势
                0.0  # 夹爪驱动关节 (4C2_Joint1) 闭合
            ]),
            pose=sapien.Pose(p=[0, 0.2, 0], q=[0, 0, 1, 0]),
        )
    )
    
    def initialize(self, engine, scene):
        """初始化机器人，保存TCP链接"""
        super().initialize(engine, scene)
        # 找到并保存TCP链接
        self._tcp_link = None
        for link in self.robot.get_links():
            if link.name == self.tcp_link_name:
                self._tcp_link = link
                break
        if self._tcp_link is None:
            raise ValueError(f"TCP link {self.tcp_link_name} not found in robot links")
    
    @property
    def tcp(self):
        """返回TCP (Tool Center Point) 链接的Actor对象"""
        if not hasattr(self, '_tcp_link') or self._tcp_link is None:
            for link in self.robot.get_links():
                if link.name == self.tcp_link_name:
                    self._tcp_link = link
                    break
        return self._tcp_link
    
    @property
    def _controller_configs(self):
        """配置机器人控制器"""
        # -------------------------------------------------------------------------- #
        # 机械臂控制器
        # -------------------------------------------------------------------------- #
        # 关节位置控制
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        
        # 关节位置增量控制 - 常用于RL
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,  # 限制每步动作幅度，增加稳定性
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        
        # 末端执行器位置增量控制
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.05,  # 减小动作空间以提高稳定性
            pos_upper=0.05,
            stiffness=self.arm_stiffness*2,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit*3,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        
        # 末端执行器姿态增量控制
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.05,  # 更小的动作空间
            pos_upper=0.05,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        
        # -------------------------------------------------------------------------- #
        # 夹爪控制器 - 使用标准PD控制器控制驱动关节，由URDF mimic处理联动
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names, # 现在只包含 ['4C2_Joint1']
            lower=0.0,  # 驱动关节的下限 (闭合)
            upper=0.82, # 驱动关节的上限 (张开)
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
            use_delta=False,
        )
        
        # 返回所有控制器配置
        controller_configs = dict(
            # 组合控制器 - 同时控制机械臂和夹爪
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
            ),
        )
        
        # 深拷贝以防用户修改
        return deepcopy(controller_configs)
    
    def is_grasping(self, obj: Actor, min_force=0.5, max_angle=85):
        """
        检查夹爪是否抓取住了物体
        
        Args:
            obj: 要检查是否被抓取的物体Actor
            min_force: 最小接触力
            max_angle: 最大接触角度（度）
            
        Returns:
            布尔值张量，表示是否抓取成功
        """
        # 简化实现：仅基于夹爪状态判断，不涉及物理接触检测
        # 检查夹爪是否闭合足够
        q = self.robot.get_qpos()
        gripper_idx = [self.robot.get_active_joints().index(j) for j in self.robot.get_active_joints() 
                      if j.name in self.gripper_joint_names[:1]]  # 只检查第一个关节
        
        if len(gripper_idx) == 0:
            return torch.zeros(self.count, dtype=torch.bool, device=self.device)
        
        gripper_pos = q[:, gripper_idx[0]]
        # 如果夹爪关闭程度小于最大开度的一半，则认为抓取成功
        return gripper_pos < 0.02
    
    def is_static(self, threshold: float = 0.1):
        """检查机器人是否静止"""
        qvel = self.robot.get_qvel()
        
        # 计算关节速度的平方和
        arm_vel = torch.zeros(self.count, device=self.device)
        for joint_name in self.arm_joint_names:
            idx = self.robot.get_qlimits().joint_map[joint_name]
            arm_vel += qvel[:, idx] ** 2
            
        is_static = arm_vel < threshold ** 2
        return is_static
    
    def get_state_names(self):
        """返回可用于状态的名称列表"""
        return ["qpos", "qvel"]


@register_agent() # 注册这个新的带相机的 Agent
class GEN72EG2RobotWristCam(GEN72EG2Robot): # 继承基础机器人
    # 新的唯一 ID
    uid = "gen72_eg2_robot_wristcam"
    urdf_path = _processed_urdf_path # 同样使用处理后的URDF路径

    @property
    def _sensor_configs(self):
        # 获取父类的传感器配置 (以防万一)
        sensor_configs = []
        if hasattr(super(), '_sensor_configs'):
             parent_configs = super()._sensor_configs
             if parent_configs:
                 sensor_configs.extend(parent_configs)

        # 添加腕部相机配置
        mount_link_name = "camera_link" # 必须与 URDF 中的名称一致
        if mount_link_name not in self.robot.links_map:
             print(f"错误：在机器人链接中找不到指定的相机挂载链接 '{mount_link_name}'！")
             print(f"可用链接: {list(self.robot.links_map.keys())}")
             raise ValueError(f"找不到相机挂载链接: {mount_link_name}")
        else:
             mount_link = self.robot.links_map[mount_link_name]
             print(f"找到相机挂载链接: {mount_link_name}")

        sensor_configs.append(
            CameraConfig(
                uid="hand_camera", # 相机传感器的唯一 ID，用于后续访问
                
                # 相机相对于 mount_link 的位姿 (Pose)
                # --- 修改: 使用单位四元数，假设 camera_link 已由 URDF 正确定向 ---
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]), 
                # --- 修改结束 --- 
                
                # --- 修改: 提高相机分辨率 ---
                width=512,      # 图像宽度 (像素)
                height=512,     # 图像高度 (像素)
                # --- 修改结束 ---
                fov=np.pi / 2, # 90 度视场角
                near=0.01,
                far=100,
                mount=mount_link, # 挂载到 camera_link 上
            )
        )
        print(f"为 {self.uid} 配置了 {len(sensor_configs)} 个传感器")
        return sensor_configs


def register_to_envs():
    """将机器人注册到相关环境中"""
    # 导入环境模块
    from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    
    environments = [PushCubeEnv, PickCubeEnv]
    robot_uid = GEN72EG2Robot.uid
    
    for env_class in environments:
        if robot_uid not in env_class.SUPPORTED_ROBOTS:
            env_class.SUPPORTED_ROBOTS.append(robot_uid)
            print(f"已将 {robot_uid} 添加到 {env_class.__name__} 的支持机器人列表中")
    
    print(f"\n现在可以在环境中使用 '{robot_uid}' 作为robot_uids参数")
    print(f"示例命令: python ppo_my.py --env_id='PushCube-v1' --robot_uids='{robot_uid}' ...")


if __name__ == "__main__":
    # 注册机器人
    robot = GEN72EG2Robot
    print(f"GEN72-EG2机器人已注册，UID: {robot.uid}")
    print(f"URDF文件路径: {_original_urdf_path}") # 显示原始URDF路径
    print(f"处理后的URDF文件路径: {_processed_urdf_path}") # 显示处理后的URDF路径
    
    # 将机器人注册到环境中
    register_to_envs() 