# 自定义机器人仿真指南

本文档提供了如何在ManiSkill中使用自定义机械臂GEN72和夹爪EG2-4C2进行仿真的指南。

## 文件说明

- `GEN72.urdf` - 机械臂的URDF文件
- `EG2-4C2.urdf` - 夹爪的URDF文件
- `../my_robot.py` - 定义自定义机器人的Python类
- `../test_my_robot.py` - 测试自定义机器人的脚本
- `../analyze_urdf.py` - 分析URDF文件的工具脚本

## 使用步骤

### 1. 分析URDF文件

首先，分析URDF文件以获取关节和链接名称：

```bash
python analyze_urdf.py my_urdf/GEN72.urdf
python analyze_urdf.py my_urdf/EG2-4C2.urdf
```

根据分析结果，更新`my_robot.py`中的关节名称和链接名称。

### 2. 根据实际URDF文件修改机器人类

打开`my_robot.py`，根据URDF分析结果修改以下内容：

- 关节名称（arm_joint_names 和 gripper_joint_names）
- 关节数量（在keyframes中的qpos数组大小）
- 夹持器链接名称（在urdf_config的link字典中）

### 3. 测试单个机器人

```bash
# 测试机械臂
python test_my_robot.py -r "my_robot" -c "pd_joint_pos"

# 测试夹爪
python test_my_robot.py -r "my_gripper" -c "pd_joint_pos"
```

### 4. 测试组合机器人

```bash
python test_my_robot.py -r "my_combined_robot" -c "pd_joint_pos"
```

### 5. 尝试随机动作或关键帧动作

```bash
# 随机动作
python test_my_robot.py -r "my_robot" -c "pd_joint_delta_pos" --random-actions

# 关键帧动作
python test_my_robot.py -r "my_robot" -c "pd_joint_pos" --keyframe-actions
```

### 6. 使用GPU加速模拟（如果支持）

```bash
python test_my_robot.py -r "my_robot" -c "pd_joint_pos" -b "gpu"
```

## 提示和注意事项

1. 如果机器人行为异常，请尝试调整控制器参数（刚度、阻尼和力限制）
2. 确保夹持器链接的摩擦力足够大，以便能够抓取物体
3. 如果模拟速度太慢，考虑简化碰撞网格或调整模拟参数
4. 对于关节范围问题，检查URDF中的限制是否与控制器配置中的一致

## 如何与环境交互

当机器人能够正常工作后，可以创建实际的任务环境，例如物体抓取或操作任务。请参考ManiSkill文档中的示例任务实现。 