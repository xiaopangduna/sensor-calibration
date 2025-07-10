import cv2
import os

# 摄像头编号，通常为 0（多个摄像头可能是 1, 2...）
camera_id = 0
cap = cv2.VideoCapture(camera_id)

# 保存参数
save_dir = "./tmp/images"
save_interval = 10  # 每 N 帧保存一次
frame_count = 0
os.makedirs(save_dir, exist_ok=True)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    frame_count += 1
    cv2.imshow("Camera", frame)

    # 保存图像
    if frame_count % save_interval == 0:
        filename = os.path.join(save_dir, f"img_{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    # 按下 q 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
