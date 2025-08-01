from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores
from pathlib import Path
import numpy as np
import cv2
import os

# 参数配置
bag_path = "/home/ubuntu/Desktop/project/2505_c50b_calibrator/datasets/250507_zhongshi_c50a_calibr/2025-05-07-16-57-04.bag"
image_topic = '111' #"/ud_camera1/image_raw"
save_dir = "/home/ubuntu/Desktop/project/2505_c50b_calibrator/datasets/250507_zhongshi_c50a_calibr/2025_05_07_16_57_04"
os.makedirs(save_dir, exist_ok=True)

# 类型解析器（针对 ROS1）
typestore = get_typestore(Stores.ROS1_NOETIC)

# 打开 .bag 文件并提取图像
with Reader(Path(bag_path)) as reader:
    connections = [x for x in reader.connections if x.topic == image_topic]
    for idx, (conn, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
        # ✅ 这里修复为 ROS1 专用反序列化
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

        if not hasattr(msg, 'height'):
            continue

        height = msg.height
        width = msg.width
        dtype = np.uint8

        # 支持 mono8, rgb8, bgr8 图像格式
        if msg.encoding == 'mono8':
            img = np.frombuffer(msg.data, dtype=dtype).reshape((height, width))
        elif msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=dtype).reshape((height, width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=dtype).reshape((height, width, 3))
        else:
            print(f"Unsupported encoding: {msg.encoding}")
            continue

        filename = os.path.join(save_dir, f"img_{idx:04d}.jpg")
        cv2.imwrite(filename, img)
        print(f"Saved: {filename}")
