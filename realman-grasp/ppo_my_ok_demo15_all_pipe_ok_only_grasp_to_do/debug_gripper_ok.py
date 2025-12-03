import sapien.core as sapien

import sapien.render

import numpy as np

import time

import os



# --- 配置 ---

GRIPPER_URDF_PATH = "/home/kewei/17robo/01mydemo/urdf_01/EG2-4C2.urdf"

# 夹爪目标 qpos: [J1, J2, J3, J4, J5, J6]

# 根据您的描述 "2正1负，然后3v3的对称控制" 和具体值

# 4C2_Joint1 (索引 0): 0.82 (+)

# 4C2_Joint2 (索引 1): 0.82 (+)

# 4C2_Joint3 (索引 2): -0.82 (-)

# 4C2_Joint4 (索引 3): -0.82 (-)

# 4C2_Joint5 (索引 4): 0.82 (+)

# 4C2_Joint6 (索引 5): 0.82 (+)

GRIPPER_OPEN_QPOS = np.array([0.8, 0.8, -0.8, -0.8, 0.8, 0.8])

GRIPPER_CLOSE_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])



# PD 控制参数

STIFFNESS = 1000  # 刚度

DAMPING = 200     # 阻尼 (从 150 增加到 250)



def main():

    print("main")

    if not os.path.exists(GRIPPER_URDF_PATH):

        print(f"错误：找不到 URDF 文件: {GRIPPER_URDF_PATH}")

        return



    print("main2")

    scene = sapien.Scene() # 尝试使用SAPIEN默认配置创建场景

    print("main3: sapien.Scene() 调用成功")

   

    # # 打印 scene 对象的方法，帮助调试 (上次已运行，暂时注释)

    # print("--- Scene object methods ---")

    # for method_name in dir(scene):

    #     if callable(getattr(scene, method_name)):

    #         print(method_name)

    # print("--- End Scene object methods ---")



    # scene.set_gravity(np.array([0, 0, 0])) # AttributeError, 暂时不设置重力

    scene.set_timestep(1 / 300.0) # Scene对象直接有set_timestep方法

    print("main1: 时间步长设置成功")



    # # 尝试通过 scene.physx_system.set_gravity (如果存在) - 上次尝试失败

    # if hasattr(scene, 'physx_system') and hasattr(scene.physx_system, 'set_gravity'):

    #     print("尝试通过 scene.physx_system.set_gravity 设置重力")

    #     scene.physx_system.set_gravity(np.array([0,0,0]))

    #     print("通过 scene.physx_system.set_gravity 设置重力成功 (可能)")

    # else:

    #     print("scene.physx_system.set_gravity 不可用")



    # # 默认的时间步长可能已经够用，或者也需要通过 physx_system 设置 - Scene有直接方法

    # if hasattr(scene, 'physx_system') and hasattr(scene.physx_system, 'set_timestep'):

    #     print("尝试通过 scene.physx_system.set_timestep 设置时间步长")

    #     scene.physx_system.set_timestep(1 / 100.0)

    #     print("通过 scene.physx_system.set_timestep 设置时间步长成功 (可能)")

    # else:

    #     print("scene.physx_system.set_timestep 不可用")





    # 添加地面 (可选，但有助于观察)

    ground_material = sapien.render.RenderMaterial() # 根据文档修正

    ground_material.base_color = np.array([0.3, 0.8, 0.3, 1]) # 当前是浅灰色

    ground_material.specular = 0.5

    scene.add_ground(altitude=0, render_material=ground_material)



    # 添加光源

    scene.set_ambient_light(np.array([0.5, 0.5, 0.5]))

    scene.add_directional_light(np.array([0, 1, -1]), np.array([0.5, 0.5, 0.5]), shadow=True)



    # 创建 SAPIEN 可视化查看器

    # viewer = sapien.Viewer() # AttributeError: module 'sapien' has no attribute 'Viewer'

    viewer = scene.create_viewer() # Scene对象有create_viewer方法

    print("Viewer 创建成功")

    viewer.set_scene(scene)

    # 设置相机初始位置和朝向

    viewer.set_camera_xyz(x=0.5, y=0, z=0.5)

    viewer.set_camera_rpy(r=0, p=-np.pi / 4, y=0)



    # 加载夹爪 URDF

    loader = scene.create_urdf_loader()

    loader.fix_root_link = True # 固定基座

   

    # 设置PD驱动的平衡位姿 (可选, 如果URDF中没有很好定义)

    # loader.set_drive_property(stiffness=STIFFNESS, damping=DAMPING, force_limit=100, mode='force')





    try:

        gripper_robot = loader.load(GRIPPER_URDF_PATH)

        print(f"成功加载 URDF: {GRIPPER_URDF_PATH}")

    except Exception as e:

        print(f"加载 URDF 失败: {e}")

        return

       

    gripper_robot.set_root_pose(sapien.Pose(p=[0, 0, 0.3])) # 将夹爪放置在空中



    active_joints = gripper_robot.get_active_joints()

    if len(active_joints) != 6:

        print(f"警告: URDF 中检测到 {len(active_joints)} 个活动关节, 但期望是 6 个。")

        print("活动关节名称:")

        for i, joint in enumerate(active_joints):

            print(f"  {i}: {joint.name}")

        # 如果关节数不匹配，后续控制可能会出错

    else:

        print("成功获取6个活动关节:")

        for i, joint in enumerate(active_joints):

            print(f"  {i}: {joint.name} (Limits: {joint.get_limits()})")

            # 尝试为每个关节设置驱动属性

            if hasattr(joint, 'set_drive_properties'):

                joint.set_drive_properties(stiffness=STIFFNESS, damping=DAMPING, force_limit=100) # force_limit可以调整

                print(f"    为关节 {joint.name} 设置了 drive properties (stiffness={STIFFNESS}, damping={DAMPING})")

            elif hasattr(joint, 'set_drive_property'): # 兼容旧API的尝试

                joint.set_drive_property(stiffness=STIFFNESS, damping=DAMPING) # 旧版API可能只需要stiffness和damping

                print(f"    为关节 {joint.name} 设置了 drive_property (stiffness={STIFFNESS}, damping={DAMPING})")

            else:

                print(f"    警告: 关节 {joint.name} 没有找到 set_drive_properties 或 set_drive_property 方法。夹爪可能不会动。")





    # # 设置关节驱动器属性 (stiffness 和 damping) - 旧的注释块

    # for joint in active_joints:

    #     # SAPIEN关节的驱动属性设置通常是针对特定模式的，如 `target` for position servo

    #     # 对于PD控制，我们直接在循环中设置驱动目标，SAPIEN的物理引擎会处理

    #     # joint.set_drive_property(stiffness=STIFFNESS, damping=DAMPING) # 旧版API

    #     # SAPIEN >= 1.0, 驱动属性在加载时或每个关节上设置

    #     pass # 现代SAPIEN中，加载器或关节对象本身设置PD参数



    is_open = False

    last_toggle_time = time.time()



    # --- Trajectory Interpolation Variables ---

    trajectory_active = False

    trajectory_start_time = 0.0

    trajectory_duration = 10  # Duration of the open/close movement in seconds (从 0.5 增加到 0.75)

    qpos_trajectory_start = np.zeros(6)

    qpos_trajectory_end = np.zeros(6)

    current_pd_target_qpos = GRIPPER_CLOSE_QPOS.copy() # Initial PD target

    # --- End Trajectory Interpolation Variables ---



    print("\n--- 控制循环开始 ---")

    # print("按 'O'键 打开夹爪, 'C'键 关闭夹爪。") # Manual control not used with auto-toggle

    print("夹爪将每2秒自动在打开和关闭状态之间切换，并使用0.5秒的轨迹插值。")

    print("按 'Q'键 退出。")

   

    # target_needs_update = True # Replaced by trajectory logic



    while not viewer.closed:

        current_time = time.time()



        # --- Automatic State Toggle & Trajectory Initiation ---

        if not trajectory_active and (current_time - last_toggle_time > 2.0):

            is_open = not is_open

           

            trajectory_active = True

            trajectory_start_time = current_time

            # Get current joint positions to start the trajectory from

            # Ensure active_joints has been populated and gripper_robot exists

            if gripper_robot and len(active_joints) == 6:

                # sapien.Articulation.get_qpos() returns positions for all active DOFs

                current_actual_qpos = gripper_robot.get_qpos()

                if current_actual_qpos is not None and len(current_actual_qpos) == len(qpos_trajectory_start):

                    qpos_trajectory_start = current_actual_qpos

                else:

                    # Fallback if get_qpos is problematic or returns unexpected format

                    print("警告:未能正确获取当前qpos, 使用上一个PD目标作为轨迹起点")

                    qpos_trajectory_start = current_pd_target_qpos.copy()

            else:

                # Fallback if robot/joints not ready

                print("警告: gripper_robot 或 active_joints 未就绪, 使用上一个PD目标作为轨迹起点")

                qpos_trajectory_start = current_pd_target_qpos.copy()



            qpos_trajectory_end = GRIPPER_OPEN_QPOS.copy() if is_open else GRIPPER_CLOSE_QPOS.copy()

           

            print(f"开始轨迹 -> {'打开' if is_open else '关闭'}. 从 {qpos_trajectory_start} 到 {qpos_trajectory_end}")
            time.sleep(3)

            last_toggle_time = current_time # Reset toggle timer



        # --- Trajectory Execution & PD Target Update ---

        if trajectory_active:

            elapsed_in_trajectory = current_time - trajectory_start_time

            if elapsed_in_trajectory >= trajectory_duration:

                current_pd_target_qpos = qpos_trajectory_end # Snap to final target

                trajectory_active = False

                # print(f"轨迹完成, 最终目标: {current_pd_target_qpos}")

            # else:

            #     alpha = elapsed_in_trajectory / trajectory_duration

            #     current_pd_target_qpos = (1 - alpha) * qpos_trajectory_start + alpha * qpos_trajectory_end
            # 在计算 alpha 后，可以对其进行平滑处理
            else:
                alpha = elapsed_in_trajectory / trajectory_duration
                # Smoothstep-like function for smoother start/end
                smooth_alpha = alpha * alpha * (3 - 2 * alpha)
                current_pd_target_qpos = (1 - smooth_alpha) * qpos_trajectory_start + smooth_alpha * qpos_trajectory_end

       

        # --- Apply PD Target to Joints ---

        if len(active_joints) == 6:

            # This print can be very verbose, uncomment for detailed debugging

            # print(f"设置PD目标 qpos: {current_pd_target_qpos}")

            for i, joint in enumerate(active_joints):

                joint.set_drive_target(current_pd_target_qpos[i])



        # --- Simulation Step & Rendering ---

        scene.step()

        scene.update_render()

        viewer.render()



    print("可视化窗口已关闭。")

    viewer.destroy() # 清理viewer

    scene = None # 清理scene

    # engine = None # Engine 已移除





if __name__ == '__main__':

    main()

