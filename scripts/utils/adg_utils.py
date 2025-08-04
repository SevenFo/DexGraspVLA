from dataclasses import dataclass
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class GraspTarget:
    """封装一个抓取目标在机器人基座坐标系下的信息"""
    # 最终手腕姿态
    translation: np.ndarray = np.array([0,0,0])  # shape (3,)
    orientation_quat: np.ndarray = np.array([1,0,0,0]) # shape (4,), [w, x, y, z] 
    width: float = 0.0 # 最终使用的抓取宽度 (m)

    # 多指手特有信息
    finger_angles: np.ndarray = np.array([]) # 例如: InspireHand的关节角度
    depth: float = 0.0 # 抓取深度
    grasp_type_name: str = "Unknown" # 抓取类型的名称
    
    # convention
    convention: str = "adg"
    
    def __repr__(self):
        return (
            f"GraspTarget:\n"
            f"  Wrist Translation (m)  : {np.round(self.translation, 4)}\n"
            f"  Wrist Orientation (wxyz): {np.round(self.orientation_quat, 4)}\n"
            f"  Finger Angles            : {self.finger_angles.astype(int)}\n"
            f"  Final Width (mm)         : {self.width * 1000:.2f}\n"
            f"  Applied Depth (mm)       : {self.depth * 1000:.2f}\n"
            f"  Grasp Type               : {self.grasp_type_name}"
            f"  Convention               : {self.convention}"
        )
        
    @property
    def rotation_matrix(self):
        return R.from_quat(self.orientation_quat[[1,2,3,0]]).as_matrix()
    
    def set_quat_from_matrix(self, matrix: np.ndarray):
        self.orientation_quat = R.from_matrix(matrix).as_quat(scalar_first=True)     
            
    
def convention_adg_to_galaxea(oritation:list[float]) -> np.ndarray:
    """ trans from adg convension to galaxea convension
        args: x y z w
        return: x y z w
    """
    r = R.from_quat(oritation)
    trans = R.from_matrix([[0,-1,0],[0,0,1],[-1,0,0]])
    trans_r =  r * trans.inv()
    return trans_r.as_quat()

def convention_galaaxea_to_adg(oritation:list[float]) -> np.ndarray:
    """ trans from galaxea convension to adg convension
        args: x y z w
        return: x y z w
    """
    r = R.from_quat(oritation)
    trans = R.from_matrix([[0, -1, 0],
                            [0, 0, 1],
                            [-1, 0, 0]])
    trans_r = r * trans
    return trans_r.as_quat()

def generate_finnal_grasp_target(self, 
                            multifinger_target: GraspTarget, 
                            two_finger_approach_target: GraspTarget,
                            approach_distance: float = 0.10,
                            arm_palm_axis_index: int = 2,
                            arm_palm_axis_inverse: bool = True):
    """
    执行一个精确的抓取和投掷动作。

    该函数假设所有输入位姿都已在机器人基座坐标系下。
    它使用 multifinger_target 来确定最终的手腕姿态和手型。
    它使用 two_finger_approach_target 来计算一个安全的线形接近和后退路径。

    Args:
        multifinger_target (GraspTarget): 包含多指手最终手腕位姿和手指角度的目标。
        two_finger_approach_target (GraspTarget): 仅用于定义接近方向的二指抓取位姿。
        approach_distance (float): 从多远开始沿直线接近目标。
        arm_palm_axis_index (int): 手臂-手心轴定义, default: 2, 可选 0:x, 1:y, 2:z。
        arm_palm_axis_inverse (bool): 手臂-手心轴定义, default: True, True 表示选定的 axis 表示的是从手心指向手臂，否则表示手臂指向手心。
    """
    # --- 1. 计算接近方向 ---
    # 从 "two_finger_approach_target" 中提取位姿，并计算其"小臂->手心"轴作为接近方向
    # 这是原代码中最关键、最巧妙的一步，我们完整保留它
    approach_rot_matrix = R.from_quat(two_finger_approach_target.orientation_quat).as_matrix()
    
    approach_vector = approach_rot_matrix[:, arm_palm_axis_index]
    approach_vector = approach_vector / np.linalg.norm(approach_vector) # 归一化
    approach_vector = -approach_vector if arm_palm_axis_inverse else approach_vector # inverse if needed

    # --- 2. 计算多指手的最终目标位姿矩阵 ---
    final_rot_matrix = R.from_quat(multifinger_target.orientation_quat).as_matrix()
    final_translation = multifinger_target.translation

    # 组合成 4x4 的齐次变换矩阵
    tcp_final_pose = np.eye(4)
    tcp_final_pose[:3, :3] = final_rot_matrix
    tcp_final_pose[:3, 3] = final_translation

    # --- 3. 根据抓取深度和接近方向，微调最终位置 ---
    # 这一步确保手掌能恰好贴合物体
    # 这里的 "+ (multifinger_target.depth + 0.014)" 可能需要根据你的手爪和实验进行微调
    # 它的含义是：从视觉检测到的抓取点，沿着接近方向“深入”一段距离
    # 以 InspireHand 为例
    # 注意: 这里的 target_gripper_pose 就是我们计算的 approach_vector
    # 使用 `copy()` 避免后续修改影响原始矩阵
    tcp_pose_adjusted = tcp_final_pose.copy()
    tcp_pose_adjusted[:3, 3] += (multifinger_target.depth + 0.014) * approach_vector

    # --- 4. 计算预备位置 (Pre-Grasp Pose) ---
    # 在调整后的最终位置上，沿着接近方向的反方向后退
    tcp_pre_pose = tcp_pose_adjusted.copy()
    tcp_pre_pose[:3, 3] -= approach_distance * approach_vector
    
    return tcp_pre_pose[:3,3].flatten(), R.from_matrix(tcp_pre_pose[:3,:3]).as_quat(scalar_first=True)

def create_inspire_grasp_target(
    tf_grasp_target: GraspTarget,
    grasp_type_id: int,
    target_depth_m: float,
    target_width_m: float,
    path_to_inspire_configs: str,
    physical_depth_offset_m: float = 0.014,
) -> GraspTarget | None:
    """
    根据二指抓取参数、抓取类型和目标宽度，计算并生成一个用于机器人执行的GraspTarget。

    这个函数是完全独立的，它封装了从理想二指抓取到物理世界手腕位姿的所有转换逻辑，
    包括宽度查找、位姿计算和物理标定补偿。
    
    注意：更改 target_width_m 会改变最终抓取的 orientation，这个特性是否需要？

    Args:
        tf_grasp_target (GraspTarget): 二指抓取目标 (base frame)。
        grasp_type_id (int): Inspire Hand的抓取类型ID (例如, 1-8)。
        target_depth_m: float 用户期望的最终抓取宽度深度 (m),
        target_width_m (float): 用户期望的最终抓取宽度 (m)。
        path_to_inspire_configs (str): 包含配置JSON文件的目录路径。
        physical_depth_offset_m (float): 一个经验性的物理标定值，用于在深度方向上
                                         补偿仿真与现实的差距。默认为0.014米。

    Returns:
        GraspTarget: 一个填充了计算后信息的GraspTarget实例，可直接用于机器人控制。
        None: 如果操作失败（例如，配置文件或关键键未找到）。
    """
    two_finger_translation = tf_grasp_target.translation.copy()
    two_finger_rotation_matrix = tf_grasp_target.rotation_matrix.copy()
    two_finger_depth = target_depth_m
    print(f"[create_inspire_grasp_target]: width:{target_width_m}:.3f, depth: {target_depth_m}:3f")
    
    # 1. --- 加载配置文件 ---
    lookup_file = os.path.join(path_to_inspire_configs, 'width_12Dangle_6Dangle.json')
    types_file = os.path.join(path_to_inspire_configs, 'grasp_types_info.json') 

    try:
        with open(lookup_file, 'r', encoding='UTF-8') as f:
            width_lookup_table = json.load(f)
        with open(types_file, 'r', encoding='UTF-8') as f:
            grasp_types_info = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR [adg_utils]: Required config file not found: {e.filename}")
        return None

    # 2. --- 查找并验证抓取类型 ---
    str_grasp_type_id = str(grasp_type_id)
    if str_grasp_type_id not in grasp_types_info:
        print(f"ERROR [adg_utils]: Grasp type ID '{grasp_type_id}' not found in grasp_types_info.json.")
        return None
    
    type_info = grasp_types_info[str_grasp_type_id]
    grasp_type_name = type_info['name']
    
    # 3. --- 宽度处理与查找 ---
    # 将目标宽度裁剪到该抓取类型的有效范围内
    min_w, max_w = type_info['width']
    clipped_width_m = np.clip(target_width_m, min_w, max_w)
    if clipped_width_m != target_width_m:
        print(f"WARNING [adg_utils]: Target width {target_width_m*1000:.1f}mm was clipped to {clipped_width_m*1000:.1f}mm for grasp type '{grasp_type_name}'.")

    # 将宽度转换为JSON查找表中的键格式（毫米，一位小数）
    width_key = str(np.round(clipped_width_m * 100, 1))

    # 鲁棒地从查找表中获取数据
    if grasp_type_name not in width_lookup_table:
        print(f"ERROR [adg_utils]: Grasp type name '{grasp_type_name}' not found in width_12Dangle_6Dangle.json.")
        return None
        
    grasp_params_for_width = width_lookup_table[grasp_type_name]
    if width_key not in grasp_params_for_width:
        print(f"WARNING [adg_utils]: Exact width key '{width_key}' not found. Finding closest available width...")
        available_widths_keys = np.array(list(grasp_params_for_width.keys()), dtype=float)
        closest_width_key_val = available_widths_keys[np.argmin(np.abs(available_widths_keys - float(width_key)))]
        width_key = str(closest_width_key_val)
        clipped_width_m = closest_width_key_val / 100.0 # 更新实际使用的宽度值
        print(f"INFO [adg_utils]: Using closest width: {width_key} ({clipped_width_m*1000:.1f}mm).")
    
    grasp_data = grasp_params_for_width[width_key]

    # 4. --- 提取手部参数和位姿偏移 ---
    offset_translation = np.array(grasp_data['translation'])
    offset_rotation = np.array(grasp_data['rotation'])
    finger_angles = np.array(grasp_data['6d'])

    # 5. --- 计算基础手腕位姿（不含深度）---
    # T_base_to_two_finger: 固定的二指抓取参考位姿
    T_base_to_two_finger = np.eye(4)
    T_base_to_two_finger[:3, :3] = two_finger_rotation_matrix
    T_base_to_two_finger[:3, 3] = two_finger_translation
    
    # T_wrist_to_two_finger: 从手腕到二指抓取中心的变换 (从JSON中查得)
    T_wrist_to_two_finger = np.eye(4)
    T_wrist_to_two_finger[:3, :3] = offset_rotation
    T_wrist_to_two_finger[:3, 3] = offset_translation
    
    # T_two_finger_to_wrist: 上述变换的逆，即从二指中心到手腕的变换
    T_two_finger_to_wrist = np.linalg.inv(T_wrist_to_two_finger)

    # T_base_to_wrist: 最终的基础手腕位姿
    T_base_to_wrist_no_depth = T_base_to_two_finger @ T_two_finger_to_wrist
    
    # 6. --- 应用深度和物理偏移 ---
    # 计算抓取方向：即二指抓取坐标系的X轴在基座坐标系下的表示
    grasp_direction_vector = T_base_to_two_finger[:3, 0] # 取旋转矩阵的第一列
    
    # 计算总的深度平移量
    total_depth_translation = (two_finger_depth + physical_depth_offset_m) * grasp_direction_vector
    
    # 将深度平移应用到基础手腕位姿的平移部分
    final_wrist_translation = T_base_to_wrist_no_depth[:3, 3] + total_depth_translation
    final_wrist_rotation_matrix = T_base_to_wrist_no_depth[:3, :3]

    # 7. --- 创建并返回最终的 GraspTarget 实例 ---
    grasp_target = GraspTarget(
        translation=final_wrist_translation,
        finger_angles=finger_angles,
        width=clipped_width_m,
        depth=two_finger_depth + physical_depth_offset_m, # 记录总的应用深度
        grasp_type_name=grasp_type_name
    )
    grasp_target.set_quat_from_matrix(final_wrist_rotation_matrix)

    return grasp_target

def get_joint_angles_from_width(
    grasp_type_id: int,
    target_width_m: float,
    width_lookup_table: dict,
    grasp_types_info: dict,
) -> tuple[np.ndarray | None, float]:
    """
    根据抓取类型和目标宽度，从配置文件中查找并返回对应的6D关节角度。

    Args:
        grasp_type_id (int): Inspire Hand的抓取类型ID (例如, 1-8)。
        target_width_m (float): 用户期望的抓取宽度 (米)。
        width_lookup_table: dict,
        grasp_types_info: dict,
        
    Returns:
        np.ndarray: 对应的6D关节角度数组，形状为(6,)。
        None: 如果操作失败（例如，配置文件或关键键未找到）。
    """    
    # 验证抓取类型ID
    
    str_grasp_type_id = str(grasp_type_id)
    if str_grasp_type_id not in grasp_types_info:
        print(f"ERROR [adg_utils]: Grasp type ID '{grasp_type_id}' not found in grasp_types_info.json.")
        return None, target_width_m

    type_info = grasp_types_info[str_grasp_type_id]
    grasp_type_name = type_info['name']
    min_w, max_w = type_info['width']

    # 裁剪目标宽度到有效范围
    clipped_width_m = np.clip(target_width_m, min_w, max_w)
    if clipped_width_m != target_width_m:
        print(f"WARNING [adg_utils]: Target width {target_width_m * 1000:.1f}mm was clipped to {clipped_width_m * 1000:.1f}mm for grasp type '{grasp_type_name}'.")

    # 将宽度转换为JSON查找表中的键格式（毫米，一位小数）
    width_key = str(np.round(clipped_width_m * 100, 1))

    # 查找对应的抓取参数
    if grasp_type_name not in width_lookup_table:
        print(f"ERROR [adg_utils]: Grasp type name '{grasp_type_name}' not found in width_12Dangle_6Dangle.json.")
        return None, target_width_m

    grasp_params_for_width = width_lookup_table[grasp_type_name]
    if width_key not in grasp_params_for_width:
        print(f"WARNING [adg_utils]: Exact width key '{width_key}' not found. Finding closest available width...")
        available_widths = np.array(list(grasp_params_for_width.keys()), dtype=float)
        closest_width_val = available_widths[np.argmin(np.abs(available_widths - float(width_key)))]
        width_key = str(closest_width_val)
        clipped_width_m = closest_width_val / 100.0  # 更新实际使用的宽度值
        print(f"INFO [adg_utils]: Using closest width: {width_key} ({clipped_width_m * 1000:.1f}mm).")

    # 获取并返回6D关节角度
    grasp_data = grasp_params_for_width[width_key]
    finger_angles = np.array(grasp_data['6d'])
    # i
    print("finger_angles", finger_angles)
    print("clipped_width_m", clipped_width_m)
    return finger_angles, clipped_width_m

def compare_grasp_calculations(
    tf_translation: np.ndarray,
    tf_rotation_matrix: np.ndarray,
    tf_depth: float,
    target_width_m: float,
    grasp_type_id: int,
    path_to_inspire_configs: str,
    physical_depth_offset_m: float = 0.014
):
    """
    一个统一的函数，结合并比较两种抓取计算逻辑。
    它会在每个关键步骤使用assert来检查结果是否一致。
    """
    print("="*80)
    print("Starting Comparison of Grasp Calculation Logic")
    print("="*80)

    # --- 1. 加载和准备通用输入数据 ---
    try:
        with open(os.path.join(path_to_inspire_configs, 'width_12Dangle_6Dangle.json'), 'r') as f:
            width_lookup_table = json.load(f)
        with open(os.path.join(path_to_inspire_configs, 'grasp_types_info.json'), 'r') as f:
            grasp_types_info = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: Config file not found: {e}")
        return

    type_info = grasp_types_info[str(grasp_type_id)]
    grasp_type_name = type_info['name']
    min_w, max_w = type_info['width']
    
    # 统一的宽度处理逻辑 (采用更鲁棒的方式)
    clipped_width_m = np.clip(target_width_m, min_w, max_w)
    width_key_f1 = str(np.round(clipped_width_m * 100, 1))
    width_key_f2 = str(np.round(clipped_width_m * 100, 1))
    
    grasp_params_for_width = width_lookup_table.get(grasp_type_name)
    if not grasp_params_for_width:
        print(f"ERROR: Grasp type '{grasp_type_name}' not in lookup table.")
        return

    # 鲁棒地获取width_key，并确保两者使用相同的key
    if width_key_f2 not in grasp_params_for_width:
        print(f"WARNING: Exact width key '{width_key_f2}' not found. Finding closest.")
        available_widths = np.array(list(grasp_params_for_width.keys()), dtype=float)
        closest_width_val = available_widths[np.argmin(np.abs(available_widths - float(width_key_f2)))]
        width_key_f1 = width_key_f2 = str(closest_width_val)
        print(f"INFO: Using closest width key: {width_key_f2}")
        
    print(f"INFO: Using Grasp Type: '{grasp_type_name}', Width Key: '{width_key_f2}'")

    # --- 2. 从JSON中提取手部偏移参数 ---
    # F1的逻辑
    offset_translation_f1 = np.array(width_lookup_table[grasp_type_name][width_key_f1]['translation'])
    offset_rotation_f1 = np.array(width_lookup_table[grasp_type_name][width_key_f1]['rotation'])
    
    # F2的逻辑
    grasp_data = width_lookup_table[grasp_type_name][width_key_f2]
    offset_translation_f2 = np.array(grasp_data['translation'])
    offset_rotation_f2 = np.array(grasp_data['rotation'])

    # 【断言 1】: 检查从JSON中查找到的原始偏移数据是否一致
    assert np.allclose(offset_translation_f1, offset_translation_f2), "FAIL: Offset translations from JSON do not match!"
    assert np.allclose(offset_rotation_f1, offset_rotation_f2), "FAIL: Offset rotations from JSON do not match!"
    print("✅ PASS: [Assert 1] JSON lookup data is consistent.")

    # --- 3. 构建变换矩阵 ---
    # T_base_to_two_finger: 理想二指抓取位姿
    T_base_to_two_finger = np.eye(4)
    T_base_to_two_finger[:3, :3] = tf_rotation_matrix
    T_base_to_two_finger[:3, 3] = tf_translation

    # T_wrist_to_two_finger: 从手腕到二指抓取中心的变换 (来自JSON)
    T_wrist_to_two_finger_f1 = np.eye(4)
    T_wrist_to_two_finger_f1[:3, :3] = offset_rotation_f1
    T_wrist_to_two_finger_f1[:3, 3] = offset_translation_f1

    T_wrist_to_two_finger_f2 = np.eye(4)
    T_wrist_to_two_finger_f2[:3, :3] = offset_rotation_f2
    T_wrist_to_two_finger_f2[:3, 3] = offset_translation_f2

    # 【断言 2】: 检查根据JSON数据构建的 T_wrist_to_two_finger 矩阵是否一致
    assert np.allclose(T_wrist_to_two_finger_f1, T_wrist_to_two_finger_f2), "FAIL: T_wrist_to_two_finger matrices do not match!"
    print("✅ PASS: [Assert 2] T_wrist_to_two_finger matrices are consistent.")

    # --- 4. 计算基础手腕位姿 (不含深度) ---
    # 这是两个函数计算逻辑的核心交汇点
    T_two_finger_to_wrist_f1 = np.linalg.inv(T_wrist_to_two_finger_f1)
    T_base_to_wrist_no_depth_f1 = T_base_to_two_finger @ T_two_finger_to_wrist_f1

    T_two_finger_to_wrist_f2 = np.linalg.inv(T_wrist_to_two_finger_f2)
    T_base_to_wrist_no_depth_f2 = T_base_to_two_finger @ T_two_finger_to_wrist_f2

    # 【断言 3】: 检查不含深度的基础手腕位姿矩阵是否完全一致
    assert np.allclose(T_base_to_wrist_no_depth_f1, T_base_to_wrist_no_depth_f2), "FAIL: Base wrist pose matrices (no depth) do not match!"
    print("✅ PASS: [Assert 3] Base wrist pose matrices (T_base_to_wrist_no_depth) are consistent.")

    # 【断言 4】: 单独检查旋转部分，这直接关系到你的问题
    rotation_f1 = T_base_to_wrist_no_depth_f1[:3, :3]
    rotation_f2 = T_base_to_wrist_no_depth_f2[:3, :3]
    assert np.allclose(rotation_f1, rotation_f2), "FAIL: ROTATION parts do not match!"
    print("✅ PASS: [Assert 4] Rotation matrices are consistent.")
    
    # --- 5. 分别计算最终结果 ---
    
    # === F1 的最终结果 (无深度调整) ===
    final_translation_f1 = T_base_to_wrist_no_depth_f1[:3, 3]
    final_rotation_f1 = T_base_to_wrist_no_depth_f1[:3, :3]
    print("\n--- Final Result from Logic 1 (graspgroupTR_2_TR) ---")
    print(f"Translation: {final_translation_f1.tolist()}")
    print(f"Rotation:\n{final_rotation_f1.tolist()}")
    print(f"Rotation:\n{R.from_matrix(final_rotation_f1).as_quat()}")

    # === F2 的最终结果 (应用深度调整) ===
    # 计算抓取方向向量 (二指TF的X轴在base下的表示)
    grasp_direction_vector = T_base_to_two_finger[:3, 0]
    # 计算总深度平移
    total_depth_translation = (tf_depth + physical_depth_offset_m) * grasp_direction_vector
    
    # 将深度平移应用到基础手腕位姿
    final_translation_f2 = T_base_to_wrist_no_depth_f2[:3, 3] + total_depth_translation
    final_rotation_f2 = T_base_to_wrist_no_depth_f2[:3, :3] # 旋转部分不受深度影响
    
    print("\n--- Final Result from Logic 2 (create_inspire_grasp_target) ---")
    print(f"Applied Depth Translation Vector: {total_depth_translation.tolist()}")
    print(f"Translation no depth: {T_base_to_wrist_no_depth_f2[:3,3].tolist()}")
    print(f"Translation: {final_translation_f2.tolist()}")
    print(f"Rotation:\n{final_rotation_f2.tolist()}")
    print(f"Rotation:\n{R.from_matrix(final_rotation_f2).as_quat()}")

    print("\n" + "="*80)
    print("Comparison Complete.")
    print("="*80)
    
    # 返回两种逻辑的结果以便进一步分析
    return {
        "logic1": {"translation": final_translation_f1, "rotation": final_rotation_f1},
        "logic2": {"translation": final_translation_f2, "rotation": final_rotation_f2}
    }
    
if __name__ == '__main__':
    
    r1 = R.from_euler('zyx', [90, 45, 30], degrees=True)
    r1 = r1.as_quat()
    assert not np.array_equal(r1, convention_galaaxea_to_adg(convention_adg_to_galaxea(r1))), f"{r1} != {convention_galaaxea_to_adg(convention_adg_to_galaxea(r1))}"
