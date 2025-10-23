"""
Script để kiểm tra environment và dependencies
"""
import sys
import os

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

# 1. Kiểm tra Python version và path
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# 2. Kiểm tra virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path:
    print(f"✅ Virtual environment active: {venv_path}")
else:
    print("❌ No virtual environment detected")

# 3. Kiểm tra các thư viện cần thiết
required_packages = [
    'tensorflow',
    'numpy',
    'matplotlib',
    'opencv-python',
    'mediapipe',
    'PIL'
]

print("\n📦 PACKAGE CHECK:")
print("-" * 40)

for package in required_packages:
    try:
        if package == 'opencv-python':
            import cv2
            print(f"✅ {package}: {cv2.__version__}")
        elif package == 'PIL':
            from PIL import Image
            print(f"✅ {package}: {Image.__version__}")
        elif package == 'tensorflow':
            import tensorflow as tf
            print(f"✅ {package}: {tf.__version__}")
        elif package == 'numpy':
            import numpy as np
            print(f"✅ {package}: {np.__version__}")
        elif package == 'matplotlib':
            import matplotlib
            print(f"✅ {package}: {matplotlib.__version__}")
        elif package == 'mediapipe':
            import mediapipe as mp
            print(f"✅ {package}: {mp.__version__}")
    except ImportError as e:
        print(f"❌ {package}: Not found - {e}")

# 4. Kiểm tra GPU
print("\n🖥️ GPU CHECK:")
print("-" * 40)
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
    else:
        print("❌ No GPU detected - using CPU")
except Exception as e:
    print(f"❌ Error checking GPU: {e}")

# 5. Kiểm tra dataset
print("\n📁 DATASET CHECK:")
print("-" * 40)
dataset_path = r'E:\Yoga_pose\DATASET\TRAIN'
if os.path.exists(dataset_path):
    print(f"✅ Dataset found: {dataset_path}")
    
    # Đếm số class và ảnh
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"   Classes: {len(classes)}")
    print(f"   Class names: {classes}")
    
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(images)
        print(f"   {class_name}: {len(images)} images")
    
    print(f"   Total images: {total_images}")
else:
    print(f"❌ Dataset not found: {dataset_path}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

# Kiểm tra xem có thiếu package nào không
missing_packages = []
for package in required_packages:
    try:
        if package == 'opencv-python':
            import cv2
        elif package == 'PIL':
            from PIL import Image
        elif package == 'tensorflow':
            import tensorflow as tf
        elif package == 'numpy':
            import numpy as np
        elif package == 'matplotlib':
            import matplotlib
        elif package == 'mediapipe':
            import mediapipe as mp
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("❌ Missing packages:")
    for package in missing_packages:
        print(f"   - {package}")
    print("\n💡 To install missing packages:")
    print("   pip install " + " ".join(missing_packages))
else:
    print("✅ All required packages are available!")
    print("\n🚀 You can now run the model comparison:")
    print("   python step_by_step_comparison.py")

print("\n" + "=" * 60)
