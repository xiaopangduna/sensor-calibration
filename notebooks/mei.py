import cv2
import numpy as np
import matplotlib.pyplot as plt



def space_to_plane_camodocal(P, params):
    """
    精确复现 CataCamera::spaceToPlane 函数.
    将一个3D点或一组3D点投影到畸变的像素坐标。

    参数:
        P (numpy.ndarray): 3D点，形状可以是 (3,) 或 (3, N)
        params (dict): 包含所有相机参数的字典
    返回:
        p (numpy.ndarray): 像素坐标，形状为 (2,) 或 (2, N)
    """
    # 提取参数
    xi = params['xi']
    k1, k2 = params['k1'], params['k2']
    p1, p2 = params['p1'], params['p2']
    gamma1, gamma2 = params['gamma1'], params['gamma2']
    u0, v0 = params['u0'], params['v0']

    # 确保P是二维数组以便于广播
    P = np.atleast_2d(P)
    if P.shape[0] == 1 and P.shape[1] == 3: # 处理单个 (1,3) 向量
        P = P.T
        
    # MEI模型的核心投影步骤
    norm = np.linalg.norm(P, axis=0)
    z_proj = P[2, :] + xi * norm
    
    # 归一化平面坐标 (无畸变)
    p_u = P[:2, :] / z_proj
    
    # 应用畸变 (径向 + 切向)
    mx2_u = p_u[0, :]**2
    my2_u = p_u[1, :]**2
    mxy_u = p_u[0, :] * p_u[1, :]
    rho2_u = mx2_u + my2_u
    
    rad_dist_u = k1 * rho2_u + k2 * rho2_u**2
    
    d_u_x = p_u[0, :] * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u)
    d_u_y = p_u[1, :] * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u)
    
    p_d = p_u + np.vstack((d_u_x, d_u_y))

    # 应用内参 (gamma, u0, v0) 得到最终像素坐标
    p_final_x = gamma1 * p_d[0, :] + u0
    p_final_y = gamma2 * p_d[1, :] + v0
    
    p_final = np.vstack((p_final_x, p_final_y))

    return p_final.squeeze()


def build_undistort_rectify_map(img_size, params, K_new, R=np.eye(3)):
    """
    精确复现 CataCamera::initUndistortRectifyMap 函数.
    构建用于 cv2.remap 的映射表。
    
    参数:
        img_size (tuple): (宽, 高)
        params (dict): 相机参数
        K_new (numpy.ndarray): 3x3 目标（针孔）相机的内参矩阵
        R (numpy.ndarray): 3x3 矫正旋转矩阵 (默认为单位阵，即只去畸变不旋转)
    返回:
        map1, map2: 用于 cv2.remap 的映射表
    """
    width, height = img_size
    
    K_new_inv = np.linalg.inv(K_new)
    R_inv = np.linalg.inv(R)

    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    uv_one = np.stack([u_coords.ravel(), v_coords.ravel(), np.ones(width*height)])
    
    P_3d = R_inv @ K_new_inv @ uv_one
    
    p_distorted = space_to_plane_camodocal(P_3d, params)
    
    map1 = p_distorted[0, :].reshape(height, width).astype(np.float32)
    map2 = p_distorted[1, :].reshape(height, width).astype(np.float32)
    
    return map1, map2

# ==============================================================================
# 2. 主程序
# ==============================================================================
def main():
    # ==========================================================
    # ===== 1. 请在这里直接填入您的 Camodocal 标定参数 =====
    # ==========================================================
    
    # 图像尺寸
    image_width = 1280
    image_height = 800
    
    # Camodocal 参数字典
    params = {
        # mirror_parameters
        'xi': 1.5,
        
        # distortion_parameters
        'k1': 0.001,
        'k2': 0.002,
        'p1': -0.0005,
        'p2': 0.0003,
        
        # projection_parameters
        'gamma1': 450.0,  # 相当于 fx
        'gamma2': 450.0,  # 相当于 fy
        'u0': 640.0,      # 相当于 cx
        'v0': 400.0       # 相当于 cy
    }
    
    # 图像尺寸元组
    img_size = (image_width, image_height)
    
    # ==========================================================
    # ===== 2. 配置输入图像路径和目标相机参数             =====
    # ==========================================================
    
    # 您的鱼眼图像路径
    IMAGE_FILE = 'path/to/your/fisyehe_image.jpg'

    # 定义去畸变后的目标相机 (理想针孔模型)
    # 您可以自由调整这里的焦距(fx, fy)和主点(cx, cy)来控制输出图像的视野和中心
    width, height = img_size
    K_new = np.array([
        [width / 4,    0,      width / 2],  # fx, 0, cx
        [   0,      height / 4, height / 2], # 0, fy, cy
        [   0,         0,         1      ]
    ])
    
    # 打印参数以供检查
    print("使用的 Camodocal 参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("\n使用的目标内参矩阵 K_new:\n", K_new)


    # --- 读取图像 ---
    original_img = cv2.imread(IMAGE_FILE)
    if original_img is None:
        print(f"错误：无法读取图像 '{IMAGE_FILE}'. 请检查路径。")
        return
    # 确保读取的图像尺寸与参数一致
    if original_img.shape[1] != image_width or original_img.shape[0] != image_height:
        print("警告：读取的图像尺寸与设置的参数不匹配！")
        print(f"参数尺寸: {image_width}x{image_height}, 实际图像尺寸: {original_img.shape[1]}x{original_img.shape[0]}")


    # --- 构建映射并执行去畸变 ---
    print("\n正在构建去畸变映射表 (这可能需要几秒钟)...")
    map1, map2 = build_undistort_rectify_map(img_size, params, K_new)
    print("映射表构建完成。")
    
    undistorted_img = cv2.remap(original_img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # --- 显示和保存结果 ---
    output_filename = 'undistorted_camodocal_result_hardcoded.jpg'
    cv2.imwrite(output_filename, undistorted_img)
    print(f"去畸变图像已保存至 '{output_filename}'")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始畸变图像 (Original Distorted)')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('去畸变后图像 (Undistorted)')
    axes[1].axis('off')
    
    plt.suptitle("Camodocal MEI 模型去畸变验证 (硬编码参数)")
    plt.show()

if __name__ == '__main__':
    main()
