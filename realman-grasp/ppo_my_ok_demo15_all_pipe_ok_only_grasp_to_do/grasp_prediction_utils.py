import numpy as np
import open3d as o3d
import transforms3d.quaternions as tq
import transforms3d.affines as ta
import sapien
import cv2 # 用于可能的图像加载
import torch
import sys
import os

# 尝试导入 GraspNet 相关模块
try:
    from graspnetAPI import GraspGroup
    GRASPNET_AVAILABLE = True
except ImportError:
    print("警告: graspnetAPI 未安装，将使用虚拟抓取位姿。")
    print("  安装方法: pip install graspnetAPI")
    GRASPNET_AVAILABLE = False

# 尝试导入 GraspNet baseline 模型
try:
    # 1. 尝试在项目根目录下寻找 graspnet-baseline
    _this_file = os.path.abspath(__file__)
    _project_root = os.path.dirname(os.path.dirname(_this_file)) # realman-grasp
    
    # 检查可能的路径
    possible_paths = [
        os.path.join(_project_root, 'graspnet_dependencies'), # 用户指定的依赖目录
        os.path.join(_project_root, 'graspnet-baseline'), # /home/vipuser/realman-grasp/graspnet-baseline
        os.path.join(os.path.dirname(_project_root), 'manipulator_grasp', 'graspnet-baseline'), # /home/vipuser/manipulator_grasp/graspnet-baseline
        os.path.join(_project_root, 'manipulator_grasp', 'graspnet-baseline') # 备用
    ]
    # possible_paths = [
    #     r"C:\Users\GC414\Desktop\graspnet-baseline",   # ← 你的真实路径，优先级最高
    #     r"C:\Users\GC414\Desktop\graspnet_baseline",  # 兼容下划线命名
    #     os.path.join(_project_root, 'graspnet-baseline'),
    #     os.path.join(_project_root, 'graspnet_baseline'),
    #     os.path.join(_project_root, 'graspnet_dependencies'),
    # ]

    _graspnet_baseline_path = None
    for path in possible_paths:
        if os.path.exists(path):
            _graspnet_baseline_path = path
            break
    
    if _graspnet_baseline_path:
        sys.path.insert(0, os.path.join(_graspnet_baseline_path, 'models'))
        sys.path.insert(0, os.path.join(_graspnet_baseline_path, 'dataset'))
        sys.path.insert(0, os.path.join(_graspnet_baseline_path, 'utils'))
        # 关键修复: 添加 pointnet2 目录到 path，以便能导入 pointnet2_modules
        sys.path.insert(0, os.path.join(_graspnet_baseline_path, 'pointnet2'))
        # 关键修复: 添加 knn 目录到 path，以便能导入 knn_modules
        sys.path.insert(0, os.path.join(_graspnet_baseline_path, 'knn'))
        
        from graspnet import GraspNet, pred_decode
        from collision_detector import ModelFreeCollisionDetector
        GRASPNET_MODEL_AVAILABLE = True
        print(f"成功导入 GraspNet baseline 模型 (路径: {_graspnet_baseline_path})")
    else:
        print(f"警告: 未找到 graspnet-baseline 目录。搜索路径: {possible_paths}")
        GRASPNET_MODEL_AVAILABLE = False
except ImportError as e:
    print(f"警告: 无法导入 GraspNet baseline 模型: {e}")
    GRASPNET_MODEL_AVAILABLE = False

# --- 辅助函数：将 SAPIEN 的 pose (p, q) 转换为 4x4 矩阵 (复用自 test_two_robot_stack.py) ---
def pose_to_transformation_matrix(p, q):
    """Converts SAPIEN pose (position p, quaternion q [w,x,y,z]) to 4x4 transformation matrix."""
    p = np.asarray(p).reshape(3)
    q = np.asarray(q).reshape(4) # [w, x, y, z]
    # Ensure quaternion is normalized (safer)
    q = q / np.linalg.norm(q)
    try:
        R = tq.quat2mat(q)
    except ValueError as e:
        print(f"Error converting quaternion {q} to matrix: {e}")
        # Return identity if quaternion is invalid
        return np.eye(4)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

# --- 点云生成 ---
def depth_to_point_cloud(depth_npy_path, intrinsics_matrix, depth_scale=1000.0, clip_distance_m=2.0):
    """
    从保存的深度图 .npy 文件和相机内参生成点云 (OpenCV 相机坐标系)。

    Args:
        depth_npy_path (str): 深度图 .npy 文件路径。
        intrinsics_matrix (np.ndarray): 3x3 相机内参矩阵 (fx, 0, cx; 0, fy, cy; 0, 0, 1)。
        depth_scale (float): 将深度图像素值转换为米所需的除数 (例如，如果深度单位是毫米，则为 1000.0)。
        clip_distance_m (float): 忽略超过此距离（米）的点。

    Returns:
        o3d.geometry.PointCloud | None: 生成的点云对象 (在相机坐标系下)，失败则为 None。
        np.ndarray | None: 点云的 NumPy 数组 (N, 3)，失败则为 None。
    """
    try:
        depth_image_raw = np.load(depth_npy_path) # 加载深度图
        
        # 处理可能的不同 dtype 和单位 (假设输入是毫米 uint16 或 float)
        if depth_image_raw.dtype == np.uint16:
             depth_image_mm = depth_image_raw
        elif depth_image_raw.dtype == np.float32 or depth_image_raw.dtype == np.float64:
             # 如果是浮点数，假设单位已经是米，需要转换回 uint16 毫米给 Open3D
             print("警告: 深度图是浮点数，假设单位是米，将转换为 uint16 毫米。")
             depth_image_mm = (depth_image_raw * 1000).astype(np.uint16)
        else:
            # 尝试强制转换为 uint16
            print(f"警告: 未知深度图 dtype ({depth_image_raw.dtype})，尝试转换为 uint16 毫米。")
            depth_image_mm = depth_image_raw.astype(np.uint16)

        # 移除可能存在的 channel 维度 (H, W, 1) -> (H, W)
        if depth_image_mm.ndim == 3 and depth_image_mm.shape[-1] == 1:
            depth_image_mm = np.squeeze(depth_image_mm, axis=-1)
        elif depth_image_mm.ndim != 2:
             raise ValueError(f"深度图维度不正确: {depth_image_mm.shape}")

        print(f"处理后的深度图 (uint16 mm): shape={depth_image_mm.shape}, dtype={depth_image_mm.dtype}, min={np.min(depth_image_mm)}, max={np.max(depth_image_mm)}")

        # 将深度图转换为 Open3D Image 对象
        o3d_depth = o3d.geometry.Image(depth_image_mm)

        # 获取图像尺寸
        height, width = depth_image_mm.shape

        # 创建 Open3D 相机内参对象
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2]
        )
        print(f"Open3D 内参: \n{o3d_intrinsics.intrinsic_matrix}")

        # 从深度图和内参创建点云
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            o3d_intrinsics,
            depth_scale=depth_scale, # 将毫米转为米
            depth_trunc=clip_distance_m, # 截断远距离点
            project_valid_depth_only=True
        )

        if not pcd.has_points():
            print("警告: 未能从深度图生成任何点。")
            return None, None

        points_np = np.asarray(pcd.points)
        print(f"生成的点云包含 {len(points_np)} 个点。")
        return pcd, points_np

    except FileNotFoundError:
        print(f"错误: 找不到深度文件 {depth_npy_path}")
        return None, None
    except Exception as e:
        import traceback
        print(f"从深度图生成点云时出错: {e}")
        print(traceback.format_exc())
        return None, None

# --- GraspNet 模型加载 (全局变量，避免重复加载) ---
_graspnet_model = None

def load_graspnet_model(checkpoint_path):
    """加载 GraspNet 模型（只加载一次）"""
    global _graspnet_model
    
    if _graspnet_model is not None:
        return _graspnet_model
    
    if not GRASPNET_MODEL_AVAILABLE:
        print("错误: GraspNet baseline 模型未安装，无法加载。")
        return None
    
    try:
        print(f"正在加载 GraspNet 模型: {checkpoint_path}")
        net = GraspNet(
            input_feature_dim=0, 
            num_view=300, 
            num_angle=12, 
            num_depth=4,
            cylinder_radius=0.05, 
            hmin=-0.02, 
            hmax_list=[0.01, 0.02, 0.03, 0.04], 
            is_training=False
        )
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            net.eval()
            print(f"成功加载 GraspNet 模型权重 (设备: {device})")
            _graspnet_model = net
            return net
        else:
            print(f"错误: 检查点文件不存在: {checkpoint_path}")
            return None
    except Exception as e:
        print(f"加载 GraspNet 模型失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# --- GraspNet 推理 ---
def get_grasp_poses_from_graspnet(rgb_image_path, points_np, intrinsics_matrix, checkpoint_path, num_point=20000, use_real_graspnet=True):
    """
    调用 GraspNet 模型进行抓取预测。

    Args:
        rgb_image_path (str): RGB 图像文件路径。
        points_np (np.ndarray): (N, 3) 点云 NumPy 数组 (相机坐标系)。
        intrinsics_matrix (np.ndarray): 3x3 相机内参矩阵。
        checkpoint_path (str): GraspNet 模型检查点路径。
        num_point (int): 采样点数。
        use_real_graspnet (bool): 是否使用真实的 GraspNet 模型。

    Returns:
        GraspGroup or list[np.ndarray]: 如果使用真实 GraspNet，返回 GraspGroup 对象；
                                        否则返回 4x4 矩阵列表。
    """
    print("--- 开始 GraspNet 推理 ---")
    print(f"  RGB 路径: {rgb_image_path}")
    print(f"  点云点数: {len(points_np) if points_np is not None else 'None'}")
    print(f"  使用真实 GraspNet: {use_real_graspnet}")

    # 如果不使用真实 GraspNet 或者依赖项未安装，返回虚拟位姿
    if not use_real_graspnet or not GRASPNET_AVAILABLE or not GRASPNET_MODEL_AVAILABLE:
        print("警告: 使用虚拟抓取位姿（GraspNet 未启用或依赖项未安装）")
        grasp_poses_T_Cam_Grasp = []
        T_Cam_Grasp_dummy1 = np.identity(4)
        T_Cam_Grasp_dummy1[2, 3] = 0.1
        grasp_poses_T_Cam_Grasp.append(T_Cam_Grasp_dummy1)
        print(f"--- GraspNet 推理结束 (返回 {len(grasp_poses_T_Cam_Grasp)} 个虚拟位姿) ---")
        return grasp_poses_T_Cam_Grasp

    try:
        # 加载模型
        net = load_graspnet_model(checkpoint_path)
        if net is None:
            print("错误: 模型加载失败，返回虚拟位姿")
            return [np.eye(4)]

        # 加载 RGB 图像
        if os.path.exists(rgb_image_path):
            color = cv2.imread(rgb_image_path)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) / 255.0
        else:
            print(f"警告: RGB 图像不存在，使用纯色")
            color = np.ones((points_np.shape[0], 3), dtype=np.float32) * 0.5

        # 采样点云
        if len(points_np) >= num_point:
            idxs = np.random.choice(len(points_np), num_point, replace=False)
        else:
            idxs1 = np.arange(len(points_np))
            idxs2 = np.random.choice(len(points_np), num_point - len(points_np), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        
        cloud_sampled = points_np[idxs]
        
        # 准备输入
        cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cloud_sampled_tensor = cloud_sampled_tensor.to(device)
        
        end_points = {'point_clouds': cloud_sampled_tensor}
        
        # 推理
        print("  正在进行 GraspNet 推理...")
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        
        # 转换为 GraspGroup
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        
        print(f"  GraspNet 预测了 {len(gg)} 个抓取位姿")
        
        # 碰撞检测
        print("  正在进行碰撞检测...")
        mfcdetector = ModelFreeCollisionDetector(points_np, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        gg = gg[~collision_mask]
        print(f"  碰撞检测后剩余 {len(gg)} 个抓取位姿")
        
        # NMS 和排序
        gg.nms()
        gg.sort_by_score()
        
        print(f"--- GraspNet 推理结束 (返回 GraspGroup，共 {len(gg)} 个位姿) ---")
        return gg
        
    except Exception as e:
        print(f"GraspNet 推理失败: {e}")
        import traceback
        print(traceback.format_exc())
        print("返回虚拟位姿")
        return [np.eye(4)]

# --- 坐标变换：获取相机在基座标系下的位姿 ---
def get_camera_pose_in_base(agent, camera_link_name="camera_link"):
    """
    计算手部相机相对于机器人基座的位姿 (T_Base_Cam)。
    依赖于 agent.robot.pose 和 camera_link.pose。

    Args:
        agent: SAPIEN agent 对象 (例如 env.agent.agents[0])。
        camera_link_name (str): 相机所在的 link 名称。

    Returns:
        np.ndarray | None: 4x4 矩阵 T_Base_Cam，如果失败则为 None。
    """
    T_World_Base = None
    T_World_Link = None
    T_Base_Cam = None

    try:
        # 1. 获取基座世界位姿 T_World_Base
        base_pose = agent.robot.pose
        p_base = base_pose.p
        q_base = base_pose.q
        if hasattr(p_base, 'cpu'): p_base = p_base.cpu().numpy()
        if hasattr(q_base, 'cpu'): q_base = q_base.cpu().numpy()
        if p_base.ndim > 1: p_base = p_base.squeeze(0)
        if q_base.ndim > 1: q_base = q_base.squeeze(0)
        T_World_Base = pose_to_transformation_matrix(p_base, q_base)
        T_Base_World = np.linalg.inv(T_World_Base) # 计算逆矩阵 T_Base_World

        # 2. 获取 camera_link 的世界位姿 T_World_Link
        links = agent.robot.get_links()
        camera_link_found = False
        for link in links:
            if link.name == camera_link_name:
                link_pose = link.pose
                p_link = link_pose.p
                q_link = link_pose.q
                if hasattr(p_link, 'cpu'): p_link = p_link.cpu().numpy()
                if hasattr(q_link, 'cpu'): q_link = q_link.cpu().numpy()
                if p_link.ndim > 1: p_link = p_link.squeeze(0)
                if q_link.ndim > 1: q_link = q_link.squeeze(0)
                T_World_Link = pose_to_transformation_matrix(p_link, q_link)
                camera_link_found = True
                break
        if not camera_link_found:
             print(f"错误: 未找到 Link '{camera_link_name}'")
             return None

        # 3. 假设相机相对于 Link 的固定变换 T_Link_cvCam
        #    (与 coordinate_transformation_tutorial.md 中的假设一致)
        T_Link_cvCam_assumed = np.array([
            [ 0.,  0.,  1.,  0.], # Link X = Cam Z
            [-1.,  0.,  0.,  0.], # Link Y = -Cam X
            [ 0., -1.,  0.,  0.], # Link Z = -Cam Y
            [ 0.,  0.,  0.,  1.]
        ])

        # 4. 计算 T_World_Cam = T_World_Link @ T_Link_cvCam_assumed
        T_World_Cam = T_World_Link @ T_Link_cvCam_assumed

        # 5. 计算 T_Base_Cam = T_Base_World @ T_World_Cam
        T_Base_Cam = T_Base_World @ T_World_Cam

        print(f"计算得到 T_Base_Cam:\n{np.round(T_Base_Cam, 3)}")
        return T_Base_Cam

    except np.linalg.LinAlgError:
        print("错误: 计算相机位姿时发生矩阵错误 (如求逆失败)。")
        return None
    except Exception as e:
        import traceback
        print(f"计算相机位姿时出错: {e}")
        print(traceback.format_exc())
        return None

# --- 可视化：使用 Open3D 绘制点云和抓取位姿 ---
def visualize_grasps(pcd, grasp_poses_or_group, intrinsics_matrix=None, T_World_Cam=None, output_filename="grasp_visualization.png", width=800, height=600, max_grasps=10):
    """
    使用 Open3D 在新弹窗中可视化点云和预测的抓取位姿。
    
    Args:
        pcd (o3d.geometry.PointCloud): Open3D 点云对象。
        grasp_poses_or_group: GraspGroup 对象 或 4x4 抓取位姿矩阵列表 (相机坐标系)。
        intrinsics_matrix (np.ndarray, optional): 3x3 相机内参矩阵 (未使用，保留接口兼容性)。
        T_World_Cam (np.ndarray, optional): 相机到世界的变换 (未使用，保留接口兼容性)。
        output_filename (str): 保存的文件名 (未使用，保留接口兼容性)。
        width (int): 窗口宽度。
        height (int): 窗口高度。
        max_grasps (int): 最多显示的抓取位姿数量。
    """
    print(f"--- 开始生成 Open3D 可视化 (新弹窗) ---")
    if pcd is None or not pcd.has_points():
        print("错误: 点云为空，无法可视化。")
        return

    # 检查点云信息
    points = np.asarray(pcd.points)
    print(f"  点云点数: {len(points)}")
    if len(points) > 0:
        print(f"  点云范围: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
              f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
              f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"  点云中心: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]")

    geometries = []
    
    # 添加相机坐标系原点 (更大的坐标轴，方便识别)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    geometries.append(camera_frame)
    print("  已添加相机坐标系原点 (大坐标轴)")
    
    # 添加点云
    geometries.append(pcd)
    
    # 处理抓取位姿
    grasp_count = 0
    
    # 检查是否是 GraspGroup 对象
    if GRASPNET_AVAILABLE and isinstance(grasp_poses_or_group, GraspGroup):
        print(f"  检测到 GraspGroup 对象，包含 {len(grasp_poses_or_group)} 个抓取")
        # 限制显示数量
        gg_to_show = grasp_poses_or_group[:max_grasps]
        print(f"  显示前 {len(gg_to_show)} 个抓取位姿")
        
        # 使用 GraspGroup 的方法生成夹爪几何体（红色夹爪）
        grippers = gg_to_show.to_open3d_geometry_list()
        geometries.extend(grippers)
        grasp_count = len(grippers)
        
        # 打印前几个抓取的位置和分数
        for i in range(min(5, len(gg_to_show))):
            pos = gg_to_show.translations[i]
            score = gg_to_show.scores[i]
            print(f"  抓取 {i+1}: 位置=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], 分数={score:.3f}")
    
    else:
        # 处理传统的 4x4 矩阵列表
        print(f"  使用传统的 4x4 矩阵列表")
        if isinstance(grasp_poses_or_group, list):
            grasp_poses_cam = grasp_poses_or_group
        else:
            grasp_poses_cam = [grasp_poses_or_group]
        
        for i, T_cam_grasp in enumerate(grasp_poses_cam):
            if i >= max_grasps:
                print(f"  (仅显示前 {max_grasps} 个抓取位姿)")
                break
            
            # 创建坐标轴 (红色=X, 绿色=Y, 蓝色=Z)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
            mesh_frame.transform(T_cam_grasp)
            geometries.append(mesh_frame)
            grasp_count += 1
            
            # 打印抓取位姿信息
            pos = T_cam_grasp[:3, 3]
            print(f"  抓取位姿 {i+1}: 位置=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # 使用 draw_geometries 在新弹窗中显示
    print(f"  显示 {len(geometries)} 个几何体 (1 个相机坐标系 + 1 个点云 + {grasp_count} 个抓取)")
    print("  请在弹出的 Open3D 窗口中查看可视化结果:")
    print("    - 大坐标轴 = 相机坐标系原点")
    if GRASPNET_AVAILABLE and isinstance(grasp_poses_or_group, GraspGroup):
        print("    - 红色夹爪 = GraspNet 预测的抓取位姿")
    else:
        print("    - 小坐标轴 = 抓取位姿")
    print("    - 点云 = 场景深度数据")
    print("  关闭窗口后程序将继续...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GraspNet 抓取位姿可视化 (相机坐标系)",
        width=width,
        height=height,
        left=50,
        top=50
    )
    print("  可视化窗口已关闭。")

# --- 将 GraspGroup 转换为 4x4 矩阵列表 ---
def graspgroup_to_matrices(gg, max_grasps=10):
    """
    将 GraspGroup 对象转换为 4x4 变换矩阵列表。
    
    Args:
        gg: GraspGroup 对象。
        max_grasps (int): 最多返回的抓取数量。
    
    Returns:
        list[np.ndarray]: 4x4 变换矩阵列表。
    """
    if not GRASPNET_AVAILABLE:
        print("错误: graspnetAPI 未安装，无法转换 GraspGroup")
        return []
    
    if not isinstance(gg, GraspGroup):
        print("警告: 输入不是 GraspGroup 对象")
        return []
    
    matrices = []
    num_grasps = min(len(gg), max_grasps)
    
    for i in range(num_grasps):
        T = np.eye(4)
        T[:3, :3] = gg.rotation_matrices[i]
        T[:3, 3] = gg.translations[i]
        matrices.append(T)
    
    return matrices

# --- 坐标变换：将相机系抓取位姿转换到基座标系 ---
def transform_grasp_to_base(grasp_pose_cam, T_Base_Cam):
    """
    将相机坐标系下的抓取位姿转换到基座标系，并应用旋转修正以匹配机械臂末端坐标系。
    
    GraspNet 输出定义: X轴=接近方向, Y轴=闭合方向, Z轴=垂直方向
    RealMan 机械臂末端定义: Z轴=接近方向
    
    因此需要将 GraspNet 的 X 轴对齐到机械臂的 Z 轴。
    这通常需要绕 Y 轴旋转 -90 度 (即 [[0,0,-1], [0,1,0], [1,0,0]])

    Args:
        grasp_pose_cam (np.ndarray): 4x4 矩阵 T_Cam_Grasp。
        T_Base_Cam (np.ndarray): 4x4 矩阵，相机相对于基座的位姿。

    Returns:
        np.ndarray: 4x4 矩阵 T_Base_Grasp。
    """
    if T_Base_Cam is None or grasp_pose_cam is None:
        print("错误: 输入 T_Base_Cam 或 grasp_pose_cam 无效，无法转换。")
        return None
    
    # 1. 计算基座下的原始抓取位姿 (GraspNet 坐标系)
    T_Base_Grasp_Raw = T_Base_Cam @ grasp_pose_cam
    
    # 2. 应用旋转修正 (X-forward -> Z-forward)
    # 绕 Grasp 坐标系的 Y 轴旋转 -90 度
    rotation_correction = np.array([
        [ 0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.],
        [ 1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  1.]
    ])
    
    # 右乘修正矩阵 (在当前 Grasp 坐标系下旋转)
    T_Base_Grasp = T_Base_Grasp_Raw @ rotation_correction
    
    print(f"转换后的抓取位姿 T_Base_Grasp (已修正坐标系):\n{np.round(T_Base_Grasp, 3)}")
    return T_Base_Grasp

# --- 将 4x4 矩阵转换为 sapien.Pose ---
def matrix_to_sapien_pose(matrix):
     """Converts 4x4 transformation matrix to sapien.Pose."""
     if matrix is None or matrix.shape != (4, 4):
         print("错误: 无效矩阵，无法转换为 sapien.Pose")
         return None
     try:
         p = matrix[:3, 3]
         q_wxyz = tq.mat2quat(matrix[:3, :3]) # 获取 wxyz 格式四元数
         # 确保四元数规范化
         q_wxyz = q_wxyz / np.linalg.norm(q_wxyz)
         return sapien.Pose(p=p, q=q_wxyz)
     except ValueError as e:
         print(f"错误: 矩阵转换为四元数失败: {e}")
         # 返回一个默认姿态或 None
         return None # 或者 sapien.Pose()
     except Exception as e:
         print(f"错误: 矩阵转 Pose 时发生未知错误: {e}")
         return None

# --- (可选) 添加可视化抓取位姿的函数 ---
# def visualize_grasp_pose(scene, T_World_Grasp, name="grasp_marker", length=0.05, radius=0.002):
#     """在 SAPIEN 场景中添加坐标轴来可视化抓取位姿。"""
#     if scene is None or T_World_Grasp is None:
#         return
#     try:
#         builder = scene.create_actor_builder()
#         # X轴 (红色)
#         builder.add_capsule_visual(radius=radius, half_length=length/2, color=[1, 0, 0, 1], pose=sapien.Pose(p=[length/2, 0, 0], q=tq.axangle2quat([0, 1, 0], np.pi/2)))
#         # Y轴 (绿色)
#         builder.add_capsule_visual(radius=radius, half_length=length/2, color=[0, 1, 0, 1], pose=sapien.Pose(p=[0, length/2, 0], q=tq.axangle2quat([1, 0, 0], -np.pi/2)))
#         # Z轴 (蓝色)
#         builder.add_capsule_visual(radius=radius, half_length=length/2, color=[0, 0, 1, 1], pose=sapien.Pose(p=[0, 0, length/2]))
        
#         grasp_marker = builder.build_static(name=name)
#         pose_world = matrix_to_sapien_pose(T_World_Grasp)
#         if pose_world:
#             grasp_marker.set_pose(pose_world)
#             print(f"已添加抓取位姿可视化标记: {name}")
#         else:
#             print(f"警告: 无法为 {name} 设置有效的世界姿态。")
#     except Exception as e:
#         print(f"添加抓取可视化时出错: {e}") 