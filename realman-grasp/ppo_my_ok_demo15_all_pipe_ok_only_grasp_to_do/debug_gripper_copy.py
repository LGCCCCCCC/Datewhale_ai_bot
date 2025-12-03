import sapien.core as sapien
import sapien.render
import numpy as np
import time
import os

# --- 配置 ---
GRIPPER_URDF_PATH = "/home/kewei/17robo/01mydemo/urdf_01/EG2-4C2.urdf"

# 原始的6关节目标，我们将基于J1的目标来计算其他关节的目标
# J1 (idx 0): master
# J2 (idx 1): mimic J1 (multiplier 1)
# J3 (idx 2): mimic J1 (multiplier -1)
# J4 (idx 3): mimic J1 (multiplier -1)
# J5 (idx 4): mimic J1 (multiplier 1)
# J6 (idx 5): mimic J1 (multiplier 1)

# 主驱动关节的目标 (只针对 4C2_Joint1)
MASTER_JOINT_NAME = "4C2_Joint1"
MASTER_JOINT_OPEN_QPOS_VAL = 0.8  # J1 打开时的值
MASTER_JOINT_CLOSE_QPOS_VAL = 0.0 # J1 关闭时的值

# PD 控制参数
# STIFFNESS_MASTER = 1000
# DAMPING_MASTER = 250 # 你可以调整这个
# FORCE_LIMIT_MASTER = 200 # 稍微调大一点试试

# # 从动关节的PD参数 (尝试与主关节相同，或稍弱)
# STIFFNESS_SLAVE = 1000 # 尝试与master相同
# DAMPING_SLAVE = 250   # 尝试与master相同
# FORCE_LIMIT_SLAVE = 100

# PD 控制参数
STIFFNESS_MASTER = 50    # 大幅降低刚度
DAMPING_MASTER = 30      # 调整阻尼以匹配低刚度，并倾向于过阻尼
FORCE_LIMIT_MASTER = 100 # 可以稍后根据需要调整

STIFFNESS_SLAVE = 50     # 与主关节一致
DAMPING_SLAVE = 30       # 与主关节一致
FORCE_LIMIT_SLAVE = 50

# 定义mimic关系 multiplier: [J2_mult, J3_mult, J4_mult, J5_mult, J6_mult]
# 对应 active_joints 中 J1 之后的关节顺序
# 这个需要根据 active_joints 的实际顺序来确定，或者按名称查找
# 假设 active_joints 的顺序是 J1, J2, J3, J4, J5, J6
MIMIC_MULTIPLIERS = {
    "4C2_Joint2": 1,
    "4C2_Joint3": -1,
    "4C2_Joint4": -1,
    "4C2_Joint5": 1,
    "4C2_Joint6": 1,
}
# 注意: URDF中所有关节都模仿J1。如果J1的索引是0，那么其他关节的目标是 J1_target * multiplier.

def main():
    print("main")
    if not os.path.exists(GRIPPER_URDF_PATH):
        print(f"错误：找不到 URDF 文件: {GRIPPER_URDF_PATH}")
        return

    scene = sapien.Scene()
    scene.set_timestep(1 / 200.0) # 保持或尝试 1/200.0
    print("时间步长设置成功")

    ground_material = sapien.render.RenderMaterial()
    ground_material.base_color = np.array([0.3, 0.8, 0.3, 1])
    scene.add_ground(altitude=0, render_material=ground_material)
    scene.set_ambient_light(np.array([0.5, 0.5, 0.5]))
    scene.add_directional_light(np.array([0, 1, -1]), np.array([0.5, 0.5, 0.5]), shadow=True)

    viewer = scene.create_viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=0.5, y=0, z=0.5)
    viewer.set_camera_rpy(r=0, p=-np.pi / 4, y=0)
    print("Viewer 创建成功")

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    try:
        gripper_robot = loader.load(GRIPPER_URDF_PATH)
        print(f"成功加载 URDF: {GRIPPER_URDF_PATH}")
    except Exception as e:
        print(f"加载 URDF 失败: {e}")
        return
    gripper_robot.set_root_pose(sapien.Pose(p=[0, 0, 0.3]))

    active_joints_list = gripper_robot.get_active_joints()
    
    # 创建一个从关节名称到关节对象的映射，并记录主关节索引
    active_joints_map = {}
    master_joint_object = None
    master_joint_idx_in_active_list = -1

    print(f"检测到 {len(active_joints_list)} 个活动关节:")
    for i, joint in enumerate(active_joints_list):
        active_joints_map[joint.name] = joint
        print(f"  {i}: {joint.name} (Limits: {joint.get_limits()})")
        if joint.name == MASTER_JOINT_NAME:
            master_joint_object = joint
            master_joint_idx_in_active_list = i # 这个索引用于从 gripper_robot.get_qpos() 中取值
            
    if master_joint_object is None:
        print(f"错误: 未找到主驱动关节 {MASTER_JOINT_NAME}")
        return
    print(f"主驱动关节 '{MASTER_JOINT_NAME}' 在活动关节列表中的索引是: {master_joint_idx_in_active_list}")

    # 设置所有活动关节的驱动属性
    for joint in active_joints_list:
        if joint.name == MASTER_JOINT_NAME:
            print(f"  为 MASTER 关节 {joint.name} 设置驱动属性 (S={STIFFNESS_MASTER}, D={DAMPING_MASTER})")
            joint.set_drive_properties(stiffness=STIFFNESS_MASTER, damping=DAMPING_MASTER, force_limit=FORCE_LIMIT_MASTER)
        elif joint.name in MIMIC_MULTIPLIERS: # 确保是从动关节
            print(f"  为 MIMIC 关节 {joint.name} 设置驱动属性 (S={STIFFNESS_SLAVE}, D={DAMPING_SLAVE})")
            joint.set_drive_properties(stiffness=STIFFNESS_SLAVE, damping=DAMPING_SLAVE, force_limit=FORCE_LIMIT_SLAVE)
        else:
            print(f"  警告: 关节 {joint.name} 未在MIMIC_MULTIPLIERS中定义，也非主关节。将使用默认(或弱)驱动属性。")
            joint.set_drive_properties(stiffness=1, damping=1, force_limit=10)


    is_open = False
    last_toggle_time = time.time()

    trajectory_active = False
    trajectory_start_time = 0.0
    trajectory_duration = 2.0 
    
    # 轨迹起点和终点现在是针对主关节的值
    qpos_master_trajectory_start = MASTER_JOINT_CLOSE_QPOS_VAL
    qpos_master_trajectory_end = MASTER_JOINT_CLOSE_QPOS_VAL
    current_master_target_qpos_val = MASTER_JOINT_CLOSE_QPOS_VAL

    print("\n--- 控制循环开始 ---")
    print(f"将计算并应用所有6个关节的目标，基于主关节'{MASTER_JOINT_NAME}'的运动和mimic关系。")
    print("按 'Q'键 退出。")
    
    # 获取所有活动关节的初始qpos
    initial_qpos_all = gripper_robot.get_qpos()
    if initial_qpos_all is None or len(initial_qpos_all) != len(active_joints_list):
        print(f"错误：无法获取所有活动关节的初始qpos或长度不匹配。")
        # 可以设置一个默认的 current_target_qpos_all，例如全0
        current_target_qpos_all = np.zeros(len(active_joints_list))
    else:
         # 初始化 current_target_qpos_all 为当前实际位置，以避免初始跳变
        current_target_qpos_all = np.array(initial_qpos_all, dtype=float)
        current_master_target_qpos_val = current_target_qpos_all[master_joint_idx_in_active_list]
        qpos_master_trajectory_start = current_master_target_qpos_val # 更新轨迹起点

    while not viewer.closed:
        current_time = time.time()

        if not trajectory_active and (current_time - last_toggle_time > 2.0): 
            is_open = not is_open
            trajectory_active = True
            trajectory_start_time = current_time
            
            all_current_qpos = gripper_robot.get_qpos()
            if all_current_qpos is not None and len(all_current_qpos) > master_joint_idx_in_active_list:
                qpos_master_trajectory_start = all_current_qpos[master_joint_idx_in_active_list]
            else:
                print(f"警告:未能正确获取主关节的当前qpos, 使用上一个PD目标作为轨迹起点")
                qpos_master_trajectory_start = current_master_target_qpos_val

            qpos_master_trajectory_end = MASTER_JOINT_OPEN_QPOS_VAL if is_open else MASTER_JOINT_CLOSE_QPOS_VAL
            print(f"主关节 '{MASTER_JOINT_NAME}' 开始轨迹 -> {'打开' if is_open else '关闭'}. 从 {qpos_master_trajectory_start:.4f} 到 {qpos_master_trajectory_end:.4f}")
            last_toggle_time = current_time 

        if trajectory_active:
            elapsed_in_trajectory = current_time - trajectory_start_time
            if elapsed_in_trajectory >= trajectory_duration:
                current_master_target_qpos_val = qpos_master_trajectory_end 
                trajectory_active = False
            else:
                alpha = elapsed_in_trajectory / trajectory_duration
                current_master_target_qpos_val = (1 - alpha) * qpos_master_trajectory_start + alpha * qpos_master_trajectory_end
        
        # 根据主关节的目标，计算所有6个关节的目标
        current_target_qpos_all[master_joint_idx_in_active_list] = current_master_target_qpos_val
        for i, joint in enumerate(active_joints_list):
            if joint.name != MASTER_JOINT_NAME and joint.name in MIMIC_MULTIPLIERS:
                multiplier = MIMIC_MULTIPLIERS[joint.name]
                current_target_qpos_all[i] = current_master_target_qpos_val * multiplier
            elif joint.name != MASTER_JOINT_NAME: # 非主也非定义的mimic，保持原位或0
                 pass # 或者 current_target_qpos_all[i] = 0 或 initial_qpos_all[i]

        # 应用目标到所有活动关节
        # gripper_robot.set_drive_target(current_target_qpos_all) # 这个是SAPIEN的API
        # 或者，如果必须单个设置：
        for i, joint in enumerate(active_joints_list):
            joint.set_drive_target(current_target_qpos_all[i])

        scene.step()
        scene.update_render()
        viewer.render()

    print("可视化窗口已关闭.")
    if viewer: 
        viewer.destroy()
    viewer = None
    scene = None

if __name__ == '__main__':
    main()