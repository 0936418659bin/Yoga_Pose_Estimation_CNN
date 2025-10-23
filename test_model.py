# ==============================
# Yoga Pose Classification Test
# ==============================

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# --- Load lại model đã train và lưu ---
MODEL_PATH = r"E:\Yoga_pose\model\Yoga-Pose-Classification-and-Skeletonization\Yoga-Pose-Classification-and-Skeletonization\models\yoga_pose_model.keras"  # đường dẫn model đã lưu
model = load_model(MODEL_PATH)
print(" Model loaded successfully!")

# --- Khai báo danh sách các lớp (labels) ---
# Hãy cập nhật đúng theo labels đã train
class_names = ["downdog", "goddess", "plank", "tree", "warrior2"]

# --- Load và xử lý ảnh test ---
# Đường dẫn đến ảnh test
TEST_IMAGE = r"E:\Yoga_pose\DATASET\TRAIN\downdog\00000128.jpg"  # đổi đường dẫn ảnh của bạn

# Load ảnh và resize giống lúc train
img = image.load_img(TEST_IMAGE, target_size=(75, 75))  # đổi kích thước đúng với model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # thêm batch dimension
img_array = img_array / 255.0  # chuẩn hóa [0,1]

# --- Dự đoán ---
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print("\n Prediction Result:")
print(f" - File: {os.path.basename(TEST_IMAGE)}")
print(f" - Pose: {predicted_class}")
print(f" - Confidence: {confidence*100:.2f}%")

# --- 5️⃣ Nếu muốn xem chi tiết từng lớp ---
print("\n Probabilities per class:")
for label, prob in zip(class_names, predictions[0]):
    print(f"   {label:10s}: {prob*100:.2f}%")
