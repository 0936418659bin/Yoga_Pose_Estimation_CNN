import cv2
import mediapipe as mp
import numpy as np
import os

# =============================
# ⚙️ Khởi tạo MediaPipe Pose
# =============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# =============================
# 📁 Đường dẫn thư mục
# =============================
input_folder = r'E:\Yoga_pose\test_mediapipe'
output_folder = r'E:\Yoga_pose\test_mediapipe\output'

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# =============================
# 🔁 Xử lý từng ảnh
# =============================
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        print(f"🔹 Processing: {image_path}")

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Không thể đọc ảnh: {filename}")
            continue

        image_height, image_width, _ = image.shape

        # Chuyển sang RGB để dùng cho MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dự đoán pose
        result = pose.process(image_rgb)

        # Ảnh nền đen
        keypoints_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Nếu phát hiện người thì vẽ khung xương
        if result.pose_landmarks:
            # Vẽ các landmarks + connections
            mp_drawing.draw_landmarks(
                keypoints_image,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
            )
        else:
            print(f"⚠️ Không phát hiện người trong ảnh: {filename}")

        # Lưu ảnh kết quả
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, keypoints_image)
        print(f"✅ Saved: {output_path}")

print("\n🎯 Processing complete.")
