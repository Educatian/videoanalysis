# Troubleshooting Guide

## 🔴 Error 1: ModuleNotFoundError: No module named 'deep_sort_realtime'

### Problem
```
ModuleNotFoundError: No module named 'deep_sort_realtime.deepsort'
```

### Cause
- `deep_sort_realtime` 패키지가 설치되지 않음
- 또는 Python 3.13 호환성 문제

### Solution ✅

#### **Option A: Use Standalone Scripts (권장)**
이미 모든 의존성 없이 작동하는 스크립트가 있습니다:

```bash
# OpenCV 기반 분석 (의존성 최소)
python analyze_video_opencv.py

# Google Colab에서 실행 (권장)
python classroom_engagement_colab.py
```

#### **Option B: Install Missing Dependencies**
```bash
pip install deep-sort-realtime
```

#### **Option C: Use Requirements File**
```bash
pip install -r requirements.txt
```

---

## 🔴 Error 2: ModuleNotFoundError: No module named 'mediapipe'

### Problem
```
ModuleNotFoundError: No module named 'mediapipe'
```

### Cause
- Python 3.13이 아직 MediaPipe를 완전히 지원하지 않음
- 또는 설치되지 않음

### Solution ✅

#### **Use OpenCV-based Script**
```bash
# MediaPipe 없이도 작동
python analyze_video_opencv.py
```

#### **Alternative: Google Colab**
Colab에서는 모든 패키지가 사전 설치됨:
- classroom_engagement_colab.py 사용

---

## 🔴 Error 3: Cannot import DeepSORTTracker

### Problem
```
ImportError: Cannot import DeepSORTTracker
```

### Cause
- `src/__init__.py`의 선택적 임포트 활성화됨
- 의존성 패키지 미설치

### Solution ✅

**Option A: Use analyze_video_opencv.py**
```bash
python analyze_video_opencv.py
```

**Option B: Full Installation**
```bash
pip install -r requirements.txt
python -c "from src import PersonDetector; print('Success')"
```

---

## 🟡 Warning: Module Loader Shows Available/Unavailable Components

### What It Means
```
╔════════════════════════════════════════════════════════════════════╗
║   Classroom Engagement Analysis System - Module Loader v1.0       ║
╚════════════════════════════════════════════════════════════════════╝

✓ Available components:
  ✓ FeatureExtractor
  ✓ EngagementClassifier

⚠ Unavailable components (missing dependencies):
  ✗ PersonDetector
  ✗ DeepSORTTracker
```

### What To Do
- 이것은 **정상**입니다
- 모든 기능이 필요하면: `pip install -r requirements.txt`
- 기본 분석만 필요하면: 스탠드얼론 스크립트 사용

---

## 🟡 Video File Not Found

### Problem
```
Error: Cannot open video: video_example.mp4
```

### Solution ✅

**Step 1: Check File Location**
```bash
# 파일이 현재 디렉토리에 있는지 확인
ls video_example.mp4
```

**Step 2: Copy File**
```bash
cp /path/to/video_example.mp4 .
```

**Step 3: Specify Full Path**
```python
# 스크립트 수정
video_path = '/absolute/path/to/video_example.mp4'
```

---

## 🟡 Memory Error or Slow Processing

### Problem
```
MemoryError: Unable to allocate ...
```

### Solution ✅

**Option 1: Reduce Frame Count**
```python
# analyze_video_opencv.py에서
analyze_video(video_path, max_frames=50)  # 100에서 50으로 감소
```

**Option 2: Use Google Colab**
- 더 많은 리소스 (12GB RAM)
- GPU 가속 옵션

**Option 3: Process in Batches**
```bash
# 여러 번에 나누어 분석
python analyze_video_opencv.py  # 100프레임
```

---

## ✅ Recommended Solutions by Use Case

### 목표: 빠른 테스트
```bash
python analyze_video_opencv.py  # 1분 안에 결과
```

### 목표: Google Colab에서 실행
```python
# Colab에 복사-붙여넣기
!git clone https://github.com/Educatian/videoanalysis.git
%cd videoanalysis
!python classroom_engagement_colab.py
```

### 목표: 모든 고급 기능 사용
```bash
# Python 3.8-3.12 사용 권장
pip install -r requirements.txt
python test_video_quick_run.py
```

### 목표: 최소 의존성
```bash
# OpenCV만 필요
pip install opencv-python numpy
python analyze_video_opencv.py
```

---

## 📊 Dependency Tree

```
┌─ analyze_video_opencv.py ✅ (No complex deps)
│  └─ opencv-python
│  └─ numpy
│
├─ classroom_engagement_colab.py ✅ (Colab included)
│  └─ opencv-python (Colab built-in)
│  └─ numpy (Colab built-in)
│
└─ src/ (Full features)
   ├─ PersonDetector → yolov7 → torch, torchvision
   ├─ DeepSORTTracker → deep-sort-realtime
   ├─ PoseEstimator → mediapipe
   ├─ FeatureExtractor → numpy, scikit-learn
   ├─ EngagementClassifier → scikit-learn, xgboost
   └─ ReportGenerator → google-generativeai
```

---

## 🔧 Verification Checklist

### Minimal Setup
- [ ] Python 3.8+
- [ ] OpenCV: `pip install opencv-python`
- [ ] NumPy: `pip install numpy`
- [ ] Test: `python analyze_video_opencv.py`

### Recommended Setup
- [ ] Python 3.8-3.12 (3.13 미지원)
- [ ] All from requirements.txt
- [ ] Test: `python test_video_quick_run.py`

### Colab Setup
- [ ] Google Account
- [ ] Access to Colab: https://colab.research.google.com/
- [ ] Video file (upload or Google Drive)
- [ ] Test: Run `classroom_engagement_colab.py`

---

## 📞 Still Having Issues?

### Check These First
1. **Python Version**: `python --version` (3.8-3.12 권장)
2. **OpenCV**: `python -c "import cv2; print(cv2.__version__)"`
3. **NumPy**: `python -c "import numpy; print(numpy.__version__)"`
4. **File Path**: 비디오 파일이 존재하는지 확인

### Quick Diagnostic
```bash
# 모든 의존성 확인
python -c "
try:
    import cv2
    import numpy
    print('✓ Basic packages OK')
except ImportError as e:
    print(f'✗ Missing: {e}')
"
```

### Nuclear Option: Fresh Install
```bash
# Python 3.10 사용 권장
pip install --upgrade pip
pip uninstall opencv-python numpy scikit-learn torch -y
pip install -r requirements.txt
```

---

## 🎯 Decision Tree

```
문제가 있나요?
│
├─ Yes → "deep_sort_realtime" 에러?
│  ├─ Yes → analyze_video_opencv.py 사용 ✅
│  └─ No → 다음 문제로
│
├─ "mediapipe" 에러?
│  ├─ Yes → Google Colab 사용 ✅
│  └─ No → 다음 문제로
│
├─ 메모리 부족?
│  ├─ Yes → max_frames=50으로 감소 ✅
│  └─ No → 다음 문제로
│
└─ 파일을 찾을 수 없음?
   ├─ Yes → 파일 경로 확인 ✅
   └─ No → 설치 가이드 참고
```

---

## 🚀 최종 권장사항

| 상황 | 해결책 |
|------|--------|
| 빠르게 테스트 | `python analyze_video_opencv.py` |
| Google Colab 사용 | `classroom_engagement_colab.py` |
| 모든 기능 필요 | `pip install -r requirements.txt` |
| Python 3.13 | Colab 권장 |
| 메모리 부족 | max_frames 감소 |
| 의존성 에러 | 스탠드얼론 스크립트 사용 |

**성공적인 분석을 기원합니다!** 🎉
