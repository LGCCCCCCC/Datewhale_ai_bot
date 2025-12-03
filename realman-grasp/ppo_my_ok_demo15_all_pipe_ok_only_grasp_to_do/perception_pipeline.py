import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import os
import time
import cv2 # 新增: 用于图像处理(腐蚀)
import open3d as o3d # 新增: 用于点云去噪

# --- 新增: 路径管理 ---
from pathlib import Path
_this_file = Path(__file__).resolve()
_this_file_dir = _this_file.parent
_project_root = _this_file_dir.parent.parent # 假设 ppo_... 在项目根目录下一层
# --- 新增结束 ---

# --- Florence-2 Imports ---
from transformers import AutoProcessor, AutoModelForCausalLM

# --- SAM Imports ---
from segment_anything import sam_model_registry, SamPredictor

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Perception Pipeline using device: {device}")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Global Variables for Models (Load once) ---
florence_model = None
florence_processor = None
sam_predictor = None

# --------------------------------------------------------------------------
# Model Loading Functions
# --------------------------------------------------------------------------

def load_florence_model(model_id="microsoft/Florence-2-large"):
    """Loads the Florence-2 model and processor.
    Returns True if successful, False otherwise.
    """
    global florence_model, florence_processor
    if florence_model is None or florence_processor is None:
        print(f"Loading Florence-2 model ({model_id})...")
        try:
            florence_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(device).eval() # Set to evaluation mode

            florence_processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print("Florence-2 model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading Florence-2 model: {e}")
            florence_model = None
            florence_processor = None
            return False
    return True

def load_sam_model(model_path, model_type="vit_h"):
    """Loads the SAM model and creates a predictor.
    Returns True if successful, False otherwise.
    """
    global sam_predictor
    if sam_predictor is None:
        print(f"Loading SAM model from {model_path} (type: {model_type})...")
        if not os.path.exists(model_path):
            print(f"Error: SAM model file not found at {model_path}")
            return False
        try:
            sam = sam_model_registry[model_type](checkpoint=model_path)
            sam.to(device=device)
            sam_predictor = SamPredictor(sam)
            print("SAM model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            sam_predictor = None
            return False
    return True

# --------------------------------------------------------------------------
# Perception Functions
# --------------------------------------------------------------------------

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

    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    full_prompt = task_prompt + text_prompt
    image_width, image_height = image_pil.size

    try:
        inputs = florence_processor(text=full_prompt, images=image_pil, return_tensors="pt").to(device, torch_dtype)

        with torch.no_grad(): # Ensure no gradients are calculated
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # print("Florence-2 Generated Text:", generated_text) # Optional debug print

        # Parse output
        loc_tokens = re.findall(r'<loc_(\d+)>', generated_text)
        detected_bboxes = []

        if len(loc_tokens) % 4 == 0 and len(loc_tokens) > 0:
            for i in range(0, len(loc_tokens), 4):
                x1_q, y1_q, x2_q, y2_q = map(int, loc_tokens[i:i+4])
                x1 = int((x1_q / 1000.0) * image_width)
                y1 = int((y1_q / 1000.0) * image_height)
                x2 = int((x2_q / 1000.0) * image_width)
                y2 = int((y2_q / 1000.0) * image_height)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_width, x2), min(image_height, y2)
                detected_bboxes.append([x1, y1, x2, y2])
        else:
            print(f"Florence-2: Could not parse location tokens ({len(loc_tokens)}) for '{text_prompt}'.")
            return None

        # Select largest bbox
        top_1_bbox = None
        max_area = -1
        if detected_bboxes:
            for bbox in detected_bboxes:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    top_1_bbox = bbox
            if top_1_bbox:
                print(f"Florence-2: Detected '{text_prompt}' (largest bbox): {top_1_bbox}")
                return top_1_bbox
            else:
                 print(f"Florence-2: Parsed boxes but failed to select largest one for '{text_prompt}'.")
                 return None
        else:
            print(f"Florence-2: No bounding boxes found for '{text_prompt}'.")
            return None

    except Exception as e:
        print(f"Error during Florence-2 detection: {e}")
        return None

def segment_object_sam(image_np, bbox):
    """Segments an object using SAM with a bounding box prompt.

    Args:
        image_np (np.ndarray): Input RGB image (H, W, 3), uint8.
        bbox (list): Bounding box [x1, y1, x2, y2].

    Returns:
        np.ndarray or None: Binary mask (H, W) or None if error.
    """
    if sam_predictor is None:
        print("Error: SAM predictor not loaded.")
        return None

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

        # masks shape: (1, H, W)
        if masks.shape[0] > 0:
            mask = masks[0] # Get the first (and only) mask
            print(f"SAM: Mask generated. Score: {scores[0]:.4f}")
            return mask # Shape (H, W), boolean
        else:
            print("SAM: Prediction returned no masks.")
            return None

    except Exception as e:
        print(f"Error during SAM segmentation: {e}")
        return None

def depth_to_point_cloud(depth, intrinsics, mask=None, rgb=None):
    """
    Converts a depth map to a point cloud, optionally masked and colored.
    Handles depth in millimeters (ManiSkill default) and converts to meters.

    Args:
        depth (np.ndarray): Depth map (H, W) or (H, W, 1), in millimeters (uint16/int16).
        intrinsics (np.ndarray): Camera intrinsic matrix (3, 3).
        mask (np.ndarray, optional): Boolean mask (H, W) to filter points.
        rgb (np.ndarray, optional): RGB image (H, W, 3), uint8.

    Returns:
        tuple: (points, colors)
            points (np.ndarray): Point cloud coordinates (N, 3) in meters.
            colors (np.ndarray or None): Point cloud colors (N, 3), uint8.
    """
    if depth.ndim == 3:
        depth = depth.squeeze(-1) # Ensure depth is (H, W)
    height, width = depth.shape

    # Get intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    # +++ DEBUG: Print parsed intrinsics +++
    print(f"  [depth_to_point_cloud] Parsed Intrinsics: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    # +++ DEBUG END +++

    # Create pixel grid
    v, u = np.indices((height, width))

    # Convert depth to meters and handle invalid values (0)
    z = depth.astype(np.float32) / 1000.0  # Convert mm to meters
    valid_depth_mask = z > 0

    # Combine with optional mask
    if mask is not None:
        valid_mask = valid_depth_mask & mask
    else:
        valid_mask = valid_depth_mask

    # +++ DEBUG: Print sample of valid (u, v, z) inputs +++
    try:
        v_valid, u_valid = np.where(valid_mask)
        z_valid = z[valid_mask]
        num_valid_points = len(z_valid)
        print(f"  [depth_to_point_cloud] Found {num_valid_points} valid points in mask.")
        if num_valid_points > 0:
            print(f"  [depth_to_point_cloud] Sample (u, v, z_meters) inputs (first 5):")
            for i in range(min(5, num_valid_points)):
                print(f"    u={u_valid[i]}, v={v_valid[i]}, z={z_valid[i]:.4f}")
        else:
            print("  [depth_to_point_cloud] No valid points found after masking.")
    except Exception as debug_e:
        print(f"  [depth_to_point_cloud] Error during debug print: {debug_e}")
    # +++ DEBUG END +++

    # Calculate x, y coordinates (only for valid points)
    x = np.zeros_like(z)
    y = np.zeros_like(z)

    x[valid_mask] = (u[valid_mask] - cx) * z[valid_mask] / fx
    y[valid_mask] = (v[valid_mask] - cy) * z[valid_mask] / fy

    points = np.stack([x, y, z], axis=-1)
    points = points[valid_mask] # Select only valid points

    colors = None
    if rgb is not None:
        assert rgb.shape[:2] == (height, width), "RGB image shape must match depth map shape"
        colors = rgb[valid_mask]

    print(f"Point Cloud: Generated {len(points)} points.")
    return points, colors

# --------------------------------------------------------------------------
# Main Pipeline Function
# --------------------------------------------------------------------------

def run_perception(rgb_image_path, depth_map_path, intrinsics, text_prompt, sam_model_path, sam_model_type="vit_h", show_visuals=False):
    """Runs the full perception pipeline: Florence-2 -> SAM -> Point Cloud Median.

    Args:
        rgb_image_path (str): Path to the RGB image file.
        depth_map_path (str): Path to the depth map (.npy file, in mm).
        intrinsics (np.ndarray): Camera intrinsic matrix (3x3).
        text_prompt (str): Text prompt for the object.
        sam_model_path (str): Path to the SAM model checkpoint.
        sam_model_type (str): Type of SAM model (e.g., "vit_h").
        show_visuals (bool): Whether to display intermediate visualizations.

    Returns:
        np.ndarray or None: Calculated center point [x, y, z] in camera frame (meters),
                             or None if any step fails.
    """
    start_time = time.time()

    # --- 1. Load Models (if not already loaded) ---
    if not load_florence_model(): return None
    if not load_sam_model(sam_model_path, sam_model_type): return None

    # --- 2. Load Input Data ---
    print("--- Loading Data ---")
    if not os.path.exists(rgb_image_path):
        print(f"Error: RGB image not found at {rgb_image_path}")
        return None
    if not os.path.exists(depth_map_path):
        print(f"Error: Depth map not found at {depth_map_path}")
        return None

    try:
        image_pil = Image.open(rgb_image_path).convert("RGB")
        image_np = np.array(image_pil) # For SAM and point cloud colors
        depth_map_mm = np.load(depth_map_path) # Assumes saved as [H, W, 1] or [H, W]
        if depth_map_mm.ndim == 3:
            depth_map_mm = depth_map_mm.squeeze(-1) # Ensure (H, W)
        print(f"Loaded RGB Image: {image_np.shape}, Depth Map: {depth_map_mm.shape} ({depth_map_mm.dtype})")
        
    except Exception as e:
        print(f"Error loading image or depth map: {e}")
        return None

    # --- 3. Object Detection (Florence-2) ---
    print("--- Running Object Detection (Florence-2) ---")
    bbox = detect_object_florence(image_pil, text_prompt)
    if bbox is None:
        print("Object detection failed.")
        return None

    # --- 4. Object Segmentation (SAM) ---
    print("--- Running Object Segmentation (SAM) ---")
    mask = segment_object_sam(image_np, bbox)
    if mask is None:
        print("Object segmentation failed.")
        return None
    
    # --- 新增: Mask 腐蚀 (Erosion) 以去除边缘噪声 ---
    try:
        print("--- Applying Mask Erosion ---")
        kernel = np.ones((5, 5), np.uint8) # 5x5 核，腐蚀程度中等
        mask_uint8 = mask.astype(np.uint8)
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
        mask = eroded_mask.astype(bool)
        print(f"Mask erosion complete. Original area: {np.sum(mask_uint8)}, Eroded area: {np.sum(mask)}")
    except Exception as e:
        print(f"Error during mask erosion: {e}")
    # --- 腐蚀结束 ---

    # --- 5. Point Cloud Generation ---
    print("--- Generating Point Cloud ---")
    
    # --- 新增: 在生成点云前过滤 Mask 内的深度值 ---
    try:
        # 复制原始深度图，避免修改原始数据
        filtered_depth_map_mm = depth_map_mm.copy()
        
        # --- 修改: 使用百分位过滤 ---
        # 定义要去除的最低和最高百分比
        lower_percentile = 10 
        upper_percentile = 90 
        
        # 获取 Mask 内的原始深度值，并去除无效值 (0)
        original_masked_depths = depth_map_mm[mask]
        valid_original_masked_depths = original_masked_depths[original_masked_depths > 0]
        
        num_original_masked = len(original_masked_depths)
        num_valid_original = len(valid_original_masked_depths)
        print(f"Depth Filtering: Mask contains {num_original_masked} pixels.")
        print(f"Depth Filtering: Found {num_valid_original} valid (non-zero) depth pixels in mask.")

        final_mask_for_pcd = np.zeros_like(mask, dtype=bool) # 初始化最终 mask

        if num_valid_original > 10: # 只有足够多的点才进行百分位过滤
            try:
                # 计算百分位阈值
                min_depth_perc = np.percentile(valid_original_masked_depths, lower_percentile)
                max_depth_perc = np.percentile(valid_original_masked_depths, upper_percentile)
                print(f"Depth Filtering: Calculated Percentiles - Lower {lower_percentile}% = {min_depth_perc:.1f}mm, Upper {upper_percentile}% = {max_depth_perc:.1f}mm")

                # 应用百分位过滤条件 (针对 Mask 内的有效深度值)
                # 这个 mask 是针对 valid_original_masked_depths 数组的
                percentile_range_mask = (valid_original_masked_depths >= min_depth_perc) & \
                                        (valid_original_masked_depths <= max_depth_perc)
                
                num_filtered = np.sum(percentile_range_mask)
                print(f"Depth Filtering: Keeping {num_filtered} pixels after percentile filtering ({lower_percentile}%-{upper_percentile}%).")

                # 创建最终用于点云的 Mask: 必须同时满足原始 Mask 为 True，深度 > 0，且在百分位范围内
                # 1. 找到原始 mask 中深度 > 0 的像素位置
                valid_depth_indices_in_original_mask = (mask & (depth_map_mm > 0))
                # 2. 在这些位置中，找到深度值在百分位范围内的像素
                final_mask_indices = (depth_map_mm[valid_depth_indices_in_original_mask] >= min_depth_perc) & \
                                     (depth_map_mm[valid_depth_indices_in_original_mask] <= max_depth_perc)
                # 3. 更新 final_mask_for_pcd: 将满足所有条件的位置设为 True
                #   我们需要一个临时的 boolean 数组来存储 final_mask_indices 的结果，它的大小应该和 np.sum(valid_depth_indices_in_original_mask) 一致
                #   然后用这个 boolean 数组来索引 valid_depth_indices_in_original_mask 为 True 的位置，并将 final_mask_for_pcd 中对应位置设为 True
                temp_bool_array_for_update = np.zeros_like(mask, dtype=bool)
                temp_bool_array_for_update[valid_depth_indices_in_original_mask] = final_mask_indices
                final_mask_for_pcd = temp_bool_array_for_update


            except Exception as e:
                print(f"Error during percentile filtering: {e}. Falling back to using only non-zero depths.")
                # 如果百分位过滤出错，则退回使用所有非零深度值
                final_mask_for_pcd = (mask & (depth_map_mm > 0))
        else:
            print(f"Depth Filtering: Not enough valid points ({num_valid_original}) for percentile filtering. Using all valid points.")
            # 如果有效点太少，也使用所有非零深度值
            final_mask_for_pcd = (mask & (depth_map_mm > 0))

        # --- 打印过滤后的 Mask 内深度值统计信息 ---
        filtered_masked_depth_values_mm = depth_map_mm[final_mask_for_pcd] # 使用过滤后的 mask 提取
        if len(filtered_masked_depth_values_mm) > 0:
            print(f"Filtered Masked Depth Stats (mm): Min={np.min(filtered_masked_depth_values_mm):.1f}, Max={np.max(filtered_masked_depth_values_mm):.1f}, Mean={np.mean(filtered_masked_depth_values_mm):.1f}, Median={np.median(filtered_masked_depth_values_mm):.1f}")
        else:
            print("Filtered mask region contains no valid depth pixels.")

    except Exception as e:
        print(f"Error during depth filtering: {e}")
        final_mask_for_pcd = mask # 如果过滤出错，则退回使用原始 mask
    # --- 新增过滤逻辑结束 ---

    # --- 修改: 使用过滤后的 Mask 生成点云 ---
    # Note: We pass the *original* depth map, the function applies the mask internally
    # object_point_cloud, _ = depth_to_point_cloud(depth_map_mm, intrinsics, mask=mask, rgb=image_np)
    object_point_cloud, _ = depth_to_point_cloud(depth_map_mm, intrinsics, mask=final_mask_for_pcd, rgb=image_np)
    # --- 修改结束 ---

    if object_point_cloud is None or len(object_point_cloud) == 0:
        print("Point cloud generation failed or resulted in zero points (after filtering).")
        return None
        
    # --- 新增: Open3D 统计离群点去除 (Statistical Outlier Removal) ---
    try:
        print("--- Running Statistical Outlier Removal ---")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_point_cloud)
        
        # 1. 统计滤波：去除稀疏噪点
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        print(f"Statistical Outlier Removal: Remaining {len(pcd.points)} points.")

        # 2. 新增: DBSCAN 聚类去噪 (去除悬浮的独立小团块)
        # eps: 邻居距离阈值 (例如 0.005m = 5mm)，如果点之间距离大于此值则视为不同簇
        # min_points: 构成一个簇的最少点数
        print("--- Running DBSCAN Clustering to remove floating noise ---")
        labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=50, print_progress=False))
        
        if len(labels) > 0:
            max_label = labels.max()
            print(f"DBSCAN found {max_label + 1} clusters.")
            
            # 统计每个簇的点数
            # labels == -1 表示噪声
            counts = np.bincount(labels[labels >= 0])
            if len(counts) > 0:
                largest_cluster_idx = np.argmax(counts)
                print(f"Keeping largest cluster (Index {largest_cluster_idx}) with {counts[largest_cluster_idx]} points.")
                
                # 只保留最大簇的点
                largest_cluster_indices = np.where(labels == largest_cluster_idx)[0]
                pcd = pcd.select_by_index(largest_cluster_indices)
            else:
                print("DBSCAN found only noise (label -1). Keeping all points as fallback.")
        
        cleaned_point_cloud = np.asarray(pcd.points)
        print(f"Final Point Cloud Size: {len(cleaned_point_cloud)}")
        
        object_point_cloud = cleaned_point_cloud
        
    except Exception as e:
        print(f"Error during outlier removal: {e}")
    # --- 去噪结束 ---

    # --- 6. Calculate Center Point (Median) ---
    print("--- Calculating Center Point (Median) ---")
    # --- Add Z-Value Statistics --- 
    try:
        z_values = object_point_cloud[:, 2]
        print(f"Point Cloud Z-Value Stats (Meters): Min={np.min(z_values):.4f}, Max={np.max(z_values):.4f}, Mean={np.mean(z_values):.4f}, Median={np.median(z_values):.4f}")
    except IndexError:
        print("Could not extract Z-values from point cloud.")
    # --- End Add Z-Value Statistics ---
    try:
        object_center_camera = np.median(object_point_cloud, axis=0)
        print(f"Calculated object center (camera frame): {object_center_camera}")
    except Exception as e:
        print(f"Error calculating median: {e}")
        return None

    # --- Optional Visualization ---
    if show_visuals:
        print("--- Displaying Visualizations ---")
        # --- 修改: 调整为 2x3 布局 ---
        plt.figure(figsize=(18, 10)) # 调整画布大小

        # --- Row 1 --- 
        # 1. Image + BBox
        plt.subplot(2, 3, 1)
        plt.imshow(image_np)
        if bbox: # 检查 bbox 是否存在
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        plt.title(f"Florence-2 ({text_prompt})")
        plt.axis('off')

        # 2. SAM Mask
        plt.subplot(2, 3, 2)
        if mask is not None: # 检查 mask 是否存在
            plt.imshow(mask, cmap='gray')
        plt.title("SAM Mask")
        plt.axis('off')

        # 3. Image + Mask Overlay
        plt.subplot(2, 3, 3)
        plt.imshow(image_np)
        if mask is not None: # 检查 mask 是否存在
            # Create a colored overlay for the mask
            masked_overlay = np.zeros_like(image_np, dtype=np.uint8)
            masked_overlay[mask] = [0, 255, 0] # Green for mask
            plt.imshow(masked_overlay, alpha=0.5) # Overlay with transparency
        plt.title("Image + Mask Overlay")
        plt.axis('off')
        
        # --- Row 2 --- 
        # 4. Filtered Depth Map + SAM Mask Overlay 
        plt.subplot(2, 3, 4)
        try:
            # 为了可视化，过滤掉过近的点 (如机械臂)
            near_filter_threshold_mm = 300 
            depth_display = depth_map_mm.astype(np.float32)
            depth_display_vis = depth_display.copy()
            depth_display_vis[depth_display_vis < near_filter_threshold_mm] = 0 # 过滤近点 (设为0)
            
            valid_depths_vis = depth_display_vis[depth_display_vis > 0] # 基于过滤后的图找有效深度
            
            if len(valid_depths_vis) > 0:
                # 使用 5% 和 95% 百分位数进行归一化以提高对比度
                min_d_vis = np.percentile(valid_depths_vis, 5)
                max_d_vis = np.percentile(valid_depths_vis, 95)
                
                # 防止 min 和 max 相等导致除零
                if max_d_vis <= min_d_vis:
                     min_d_vis = np.min(valid_depths_vis) # Fallback to min/max if percentiles are equal
                     max_d_vis = np.max(valid_depths_vis)

                # 归一化深度值到 [0, 1]
                # 先将超出 percentile 范围的值 clip 到边界
                depth_display_clipped = np.clip(depth_display_vis, min_d_vis, max_d_vis)
                
                # 执行归一化 (确保分母不为零)
                if max_d_vis > min_d_vis:
                    depth_display_normalized = (depth_display_clipped - min_d_vis) / (max_d_vis - min_d_vis)
                else:
                    depth_display_normalized = np.zeros_like(depth_display_vis) # 如果范围为0，则全黑

                depth_display_normalized[depth_display_vis <= 0] = 0 # 无效深度设为0 (黑色)
                
                # 使用更适合深度图的 colormap
                plt.imshow(depth_display_normalized, cmap='viridis')
                plt.title(f"Depth Map (Norm: {min_d_vis:.0f}-{max_d_vis:.0f}mm, Near < {near_filter_threshold_mm}mm filtered)")
                plt.colorbar()
                
                # 在过滤后的深度图上叠加 SAM Mask (红色半透明)
                if mask is not None:
                    mask_overlay_vis = np.zeros((*mask.shape, 4), dtype=np.float32) # RGBA
                    mask_overlay_vis[mask] = [1, 0, 0, 0.5] # Red, 50% alpha
                    plt.imshow(mask_overlay_vis)
                
        except Exception as e:
            print(f"Error during depth map visualization: {e}")
            plt.title("Depth Map (Error)")
        plt.axis('off')
        
        # 5. Masked Depth used for Point Cloud Calculation
        plt.subplot(2, 3, 5)
        try:
            # 创建一个只包含最终用于计算点云的深度值的图像
            masked_depth_for_pcd_vis = depth_map_mm.copy().astype(np.float32)
            masked_depth_for_pcd_vis[~final_mask_for_pcd] = 0 # 将未使用的区域设为 0
            
            valid_masked_depths_vis = masked_depth_for_pcd_vis[final_mask_for_pcd]
            
            if len(valid_masked_depths_vis) > 0:
                # 对有效区域进行归一化
                min_d_pcd_vis = np.percentile(valid_masked_depths_vis, 5)
                max_d_pcd_vis = np.percentile(valid_masked_depths_vis, 95)
                if max_d_pcd_vis <= min_d_pcd_vis:
                    min_d_pcd_vis = np.min(valid_masked_depths_vis)
                    max_d_pcd_vis = np.max(valid_masked_depths_vis)
                
                masked_depth_pcd_clipped = np.clip(masked_depth_for_pcd_vis, min_d_pcd_vis, max_d_pcd_vis)
                if max_d_pcd_vis > min_d_pcd_vis:
                    masked_depth_pcd_normalized = (masked_depth_pcd_clipped - min_d_pcd_vis) / (max_d_pcd_vis - min_d_pcd_vis)
                else:
                    masked_depth_pcd_normalized = np.zeros_like(masked_depth_for_pcd_vis)
                
                masked_depth_pcd_normalized[~final_mask_for_pcd] = 0 # 确保掩码外是黑的
                
                plt.imshow(masked_depth_pcd_normalized, cmap='viridis')
                plt.title(f"Masked Depth for PCD (Norm: {min_d_pcd_vis:.0f}-{max_d_pcd_vis:.0f}mm)")
                plt.colorbar()
            else:
                plt.imshow(np.zeros_like(masked_depth_for_pcd_vis), cmap='gray')
                plt.title("Masked Depth for PCD (No Valid Data)")
                
        except Exception as e:
            print(f"Error during masked depth visualization: {e}")
            plt.title("Masked Depth for PCD (Error)")
        plt.axis('off')
        
        # 6. Empty subplot (placeholder or for future stats)
        plt.subplot(2, 3, 6)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    end_time = time.time()
    print(f"Perception pipeline finished in {end_time - start_time:.2f} seconds.")
    return object_center_camera, object_point_cloud, final_mask_for_pcd

# --- Example Usage (for testing this script directly) ---
if __name__ == "__main__":
    print("Running perception pipeline example...")

    # --- Configuration (Adjust these paths and values) ---
    # Use the images saved by test_two_robot_stack.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "capture_images")
    rgb_path = os.path.join(image_dir, "left_arm_capture.png")
    depth_path = os.path.join(image_dir, "left_arm_depth.npy")
    sam_ckpt_path = str(_project_root / "checkpoints/sam_vit_h_4b8939.pth")
    object_name = "the banana"

    # Example Intrinsics (Replace with actual intrinsics from your run)
    # These might be printed by test_two_robot_stack.py
    # Make sure it's a 3x3 NumPy array
    example_intrinsics = np.array([
        [618.0,   0. , 319.5], # fx, 0, cx
        [  0. , 618.0, 239.5], # 0, fy, cy
        [  0. ,   0. ,   1. ]  # 0, 0, 1
    ])
    print("\nWARNING: Using EXAMPLE intrinsics. Replace with actual values from the run.")

    # --- Run the pipeline ---
    center_point = run_perception(
        rgb_image_path=rgb_path,
        depth_map_path=depth_path,
        intrinsics=example_intrinsics, # Use the example or load actual intrinsics
        text_prompt=object_name,
        sam_model_path=sam_ckpt_path,
        show_visuals=True # Set to True to see intermediate results
    )

    if center_point is not None:
        print(f"\nExample Result: Calculated center point for '{object_name}' (camera frame): {center_point}")
    else:
        print("\nExample Result: Perception pipeline failed.") 