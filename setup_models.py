# -*- coding: utf-8 -*-
"""
자동 모델 다운로드 및 설정
YOLOv7, MediaPipe 등 필요한 모델 자동 설치
"""

import os
import sys
from pathlib import Path
import urllib.request
import json

print("=" * 80)
print("MODEL SETUP - Automatic Download & Installation")
print("=" * 80)

# 1. 디렉토리 생성
print("\n[1/5] Creating directories...")
dirs = [
    "models",
    "outputs/reports",
    "outputs/videos",
    "data/cache"
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  OK {d}/")

# 2. 필수 패키지 확인
print("\n[2/5] Checking required packages...")

packages_check = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "yaml": "pyyaml",
}

missing = []
for module_name, pkg_name in packages_check.items():
    try:
        __import__(module_name)
        print(f"  OK {pkg_name}")
    except ImportError:
        print(f"  X {pkg_name} (missing)")
        missing.append(pkg_name)

if missing:
    print(f"\nMissing: {', '.join(missing)}")
    print("Run: pip install " + " ".join(missing))

# 3. YOLOv7 모델 다운로드
print("\n[3/5] Setting up YOLOv7 model...")

yolo_path = Path("models/yolov7.pt")
if yolo_path.exists():
    print(f"  OK YOLOv7 already downloaded ({yolo_path.stat().st_size / 1e6:.0f}MB)")
else:
    print("  Downloading YOLOv7 (this may take a few minutes)...")
    try:
        url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        print(f"  From: {url}")
        
        # 다운로드 진행 표시
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}% ({downloaded/1e6:.0f}MB/{total_size/1e6:.0f}MB)", end="")
        
        urllib.request.urlretrieve(url, yolo_path, download_progress)
        print(f"\n  OK YOLOv7 downloaded ({yolo_path.stat().st_size / 1e6:.0f}MB)")
    except Exception as e:
        print(f"\n  X Download failed: {e}")
        print("  Using OpenCV-only mode")

# 4. MediaPipe 확인
print("\n[4/5] Checking MediaPipe...")

mediapipe_ok = False
try:
    import mediapipe
    print(f"  OK MediaPipe {mediapipe.__version__}")
    mediapipe_ok = True
except ImportError:
    print("  X MediaPipe not installed")
    print("  Run: pip install mediapipe")

# 5. 구성 파일 생성
print("\n[5/5] Creating configuration...")

config = {
    "system": {
        "mode": "full" if mediapipe_ok and yolo_path.exists() else "opencv",
        "models_available": {
            "yolov7": yolo_path.exists(),
            "mediapipe": mediapipe_ok,
            "deepsort": False  # 자동 대체됨
        }
    },
    "models": {
        "yolov7": str(yolo_path),
        "confidence_threshold": 0.45
    },
    "output": {
        "reports_dir": "outputs/reports",
        "videos_dir": "outputs/videos",
        "cache_dir": "data/cache"
    }
}

config_file = Path("config/setup_status.json")
config_file.parent.mkdir(parents=True, exist_ok=True)
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"  OK Configuration saved to {config_file}")

# 최종 상태
print("\n" + "=" * 80)
print("SETUP STATUS")
print("=" * 80)

mode = config["system"]["mode"]
print(f"\nMode: {mode.upper()}")

if mode == "full":
    print("\n✓ Full feature set available")
    print("  Run: python comprehensive_analysis.py")
else:
    print("\n✓ OpenCV-only mode active")
    print("  Run: python run_simple_analysis.py")

print("\nAvailable models:")
for model, available in config["system"]["models_available"].items():
    status = "OK" if available else "X"
    print(f"  [{status}] {model}")

print("\n" + "=" * 80)
