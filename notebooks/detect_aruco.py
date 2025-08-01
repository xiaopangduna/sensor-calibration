import cv2
import os
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm



# 设置参数
# image_folder = "/home/ubuntu/Desktop/project/2504_c50a_calibrator/dataset/250701_zhongshi_c50a_test_calib_board/2025-07-01"  # 替换为你的图片文件夹路径
image_folder= "/home/ubuntu/Desktop/project/utils_python/tmp/2025-07-28-17-01-49/nuwa_1_rgb0_image"
aruco_dict_type = cv2.aruco.DICT_4X4_50
save_vis = True
output_folder = os.path.join(image_folder, "output")

# 创建输出文件夹
if save_vis and not os.path.exists(output_folder):
    os.makedirs(output_folder)
duplicate_id_images = []  # 记录哪些图片有重复ID
# 加载 ArUco 字典和检测参数
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
parameters = cv2.aruco.DetectorParameters()

# 初始化统计变量
id_counts = defaultdict(int)
id_positions = defaultdict(list)
num_images = 0
num_detected = 0
parameters.useAruco3Detection = True

# parameters.aprilTagMaxLineFitMse = 0.0001
# parameters.minMarkerDistanceRate = 0.02
# parameters.maxMarkerPerimeterRate = 3.0
# parameters.adaptiveThreshConstant = 30
# parameters.adaptiveThreshWinSizeMax = 12
# parameters.adaptiveThreshWinSizeMin = 10
# parameters.adaptiveThreshWinSizeStep = 3
# parameters.polygonalApproxAccuracyRate = 0.04

# parameters.aprilTagCriticalRad = 100 没用
# parameters.aprilTagMaxNmaxima = 1000
# parameters.aprilTagMinClusterPixels = 100000000 没用
# parameters.aprilTagMinWhiteBlackDiff = 100000 没用
# parameters.aprilTagQuadSigma = 100.0  没用
# 遍历图片文件
image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

for filename in tqdm(image_files, desc="Processing images"):
    filepath = os.path.join(image_folder, filename)
    # filepath = "/home/ubuntu/Desktop/project/2504_c50a_calibrator/dataset/250701_zhongshi_c50a_test_calib_board/2025-07-01/front_176.jpg"
    image = cv2.imread(filepath)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    num_images += 1
    if ids is not None:
        num_detected += 1
        # ✅ 检查重复 ID
        id_list = ids.flatten().tolist()
        id_counter = Counter(id_list)
        repeated_ids = [k for k, v in id_counter.items() if v > 1]
        if repeated_ids:
            duplicate_id_images.append((filename, repeated_ids))

        for i, marker_id in enumerate(ids.flatten()):
            id_counts[marker_id] += 1
            id_positions[marker_id].append(corners[i][0])  # shape (4, 2)
        if save_vis:
            vis_img = image.copy()
            cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
            cv2.imwrite(os.path.join(output_folder, filename), vis_img)
    # break

# 打印统计结果
print("\n=== ArUco 检测统计结果 ===")
print(f"总图像数量: {num_images}")
print(f"成功检测数量: {num_detected}")
print(f"平均检测率: {num_detected / num_images * 100:.2f}%")
print("\n每个 Marker 的检测频率和稳定性:")

# for marker_id in sorted(id_counts):
#     positions = np.array(id_positions[marker_id])
#     mean_pos = np.mean(positions, axis=0)
#     std_dev = np.std(positions, axis=0)
#     print(f"- ID {marker_id}: 出现 {id_counts[marker_id]} 次 | 平均位置: {mean_pos.round(1)} | 标准差: {std_dev.round(2)}")

print("\n每个 Marker 的检测频率和角点标准差:")
for marker_id in sorted(id_counts):
    positions = np.array(id_positions[marker_id])  # shape = (N, 4, 2)
    corner_std = np.std(positions, axis=0)         # shape = (4, 2)，每个角点的 std
    mean_std = np.mean(corner_std, axis=0)         # shape = (2,)，平均 std
    print(f"- ID {marker_id}: 出现 {id_counts[marker_id]} 次 | 角点标准差 = {mean_std.round(2).tolist()}")

print("\n=== 重复 ID 检测报告 ===")
if duplicate_id_images:
    print(f"共有 {len(duplicate_id_images)} 张图片检测到重复 ID：")
    for filename, repeated_ids in duplicate_id_images:
        print(f"- {filename}: 重复 ID = {repeated_ids}")
else:
    print("未检测到任何重复 ID")
