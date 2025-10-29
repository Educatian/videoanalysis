# 비디오 테스트 가이드 (Video Testing Guide)

## 1. 준비 단계 (Setup)

### 1.1 테스트 비디오 준비
```bash
# 프로젝트 디렉토리에서:
mkdir -p data/raw
mkdir -p outputs/reports
mkdir -p outputs/visualizations

# video_example.mp4를 data/raw/에 복사
cp /path/to/video_example.mp4 data/raw/video_example.mp4

# 확인
ls -lh data/raw/video_example.mp4
```

### 1.2 비디오 정보 확인
```bash
# ffmpeg 설치 (if not already installed)
pip install moviepy

# 또는 Python으로 비디오 정보 확인:
python -c "
import cv2
cap = cv2.VideoCapture('data/raw/video_example.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps

print(f'비디오 정보:')
print(f'  해상도: {width}x{height}')
print(f'  FPS: {fps}')
print(f'  총 프레임: {total_frames}')
print(f'  재생시간: {duration_sec:.1f}초 ({duration_sec/60:.1f}분)')
"
```

---

## 2. 빠른 테스트 (Quick Test - 첫 30프레임만)

### 2.1 테스트 스크립트 생성
```bash
cat > test_video_quick.py << 'EOF'
"""
빠른 테스트: 비디오의 첫 30프레임만 처리
"""

import yaml
import cv2
import numpy as np
import sys
from pathlib import Path

# 설정 로드
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# 모듈 임포트
from src import (
    PersonDetector, DeepSORTTracker, PoseEstimator,
    FeatureExtractor, EngagementClassifier
)

print("=" * 80)
print("QUICK VIDEO TEST - 첫 30프레임 분석")
print("=" * 80)

# 1. 컴포넌트 초기화
print("\n[1/4] 컴포넌트 초기화 중...")
try:
    detector = PersonDetector(
        model_path='models/yolov7.pt',
        confidence_threshold=0.45,
        device='cpu',  # GPU가 있으면 'cuda'로 변경
        half_precision=False,
    )
    print("  ✓ YOLOv7 Detector 준비됨")
except Exception as e:
    print(f"  ✗ Detector 초기화 실패: {e}")
    sys.exit(1)

try:
    tracker = DeepSORTTracker(max_age=30, min_hits=3)
    print("  ✓ DeepSORT Tracker 준비됨")
except Exception as e:
    print(f"  ✗ Tracker 초기화 실패: {e}")

try:
    pose_estimator = PoseEstimator(
        model_complexity=0,  # Lite model
        enable_hand_tracking=True,
    )
    print("  ✓ MediaPipe Pose 준비됨")
except Exception as e:
    print(f"  ✗ Pose estimator 초기화 실패: {e}")
    sys.exit(1)

feature_extractor = FeatureExtractor(config)
classifier = EngagementClassifier(config)
print("  ✓ Feature Extractor & Classifier 준비됨")

# 2. 비디오 로드
print("\n[2/4] 비디오 로드 중...")
video_path = 'data/raw/video_example.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"  ✗ 비디오를 열 수 없음: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"  ✓ 비디오 로드됨:")
print(f"    - 해상도: {width}x{height}")
print(f"    - FPS: {fps}")
print(f"    - 총 프레임: {total_frames}")

# 3. 프레임 처리
print("\n[3/4] 프레임 처리 중 (최대 30프레임)...")
frame_idx = 0
max_frames = min(30, total_frames)
results = []

while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # 감지 (Detection)
        detections = detector.detect(frame)
        
        # 추적 (Tracking)
        tracked_objects = tracker.update(detections, frame_idx, frame)
        
        # 자세 추정 (Pose Estimation)
        pose_results = {}
        for obj in tracked_objects:
            track_id = obj['track_id']
            x_min, y_min, x_max, y_max = obj['bbox']
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                pose_results[track_id] = pose_estimator.estimate(cropped)
        
        # 특징 추출 (Feature Extraction)
        timestamp = frame_idx / fps
        frame_features = feature_extractor.extract_per_frame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            tracked_objects=tracked_objects,
            pose_results=pose_results,
            fps=fps,
        )
        
        # 분류 (Classification)
        frame_classifications = {}
        for student_id, features in frame_features.items():
            classification = classifier.classify(features)
            frame_classifications[student_id] = classification
        
        results.append({
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'num_detections': len(tracked_objects),
            'classifications': frame_classifications,
        })
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  ✓ 프레임 {frame_idx + 1}/{max_frames} 처리됨 "
                  f"({len(tracked_objects)} 학생 감지)")
        
        frame_idx += 1
        
    except Exception as e:
        print(f"  ⚠ 프레임 {frame_idx} 처리 중 오류: {e}")
        frame_idx += 1
        continue

cap.release()

# 4. 결과 분석
print("\n[4/4] 결과 분석 중...")
print("\n" + "=" * 80)
print("분석 결과")
print("=" * 80)

if not results:
    print("결과 없음 - 비디오에 인식된 사람이 없을 수 있습니다")
    sys.exit(0)

# 학생별 통계
all_students = set()
for result in results:
    for student_id in result['classifications'].keys():
        all_students.add(student_id)

for student_id in sorted(all_students):
    print(f"\n【{student_id}】")
    scores = []
    
    for result in results:
        if student_id in result['classifications']:
            score = result['classifications'][student_id]['engagement_score']
            level = result['classifications'][student_id]['engagement_level']
            scores.append(score)
            
            # 첫 번째와 마지막 프레임 상세 정보 출력
            if result['frame_idx'] == 0 or result['frame_idx'] == len(results) - 1:
                print(f"  [Frame {result['frame_idx']}] 참여도: {score:.1f}/100 ({level})")
                
                features = result['classifications'][student_id]['feature_values']
                print(f"    • 시선: {features.get('gaze_mean_deg', 'N/A'):.1f}°")
                print(f"    • 자세 안정성: ±{features.get('posture_stability_std_deg', 'N/A'):.1f}°")
                print(f"    • 손 활동: {features.get('gesture_frequency_permin', 'N/A'):.1f} events/min")
    
    if scores:
        print(f"  평균 참여도: {np.mean(scores):.1f}/100 "
              f"(±{np.std(scores):.1f})")

print("\n" + "=" * 80)
print("✓ 테스트 완료!")
print("=" * 80)

EOF

python test_video_quick.py
```

---

## 3. 전체 비디오 분석 (Full Analysis)

### 3.1 전체 분석 스크립트
```bash
cat > analyze_video_full.py << 'EOF'
"""
전체 비디오 분석 및 리포트 생성
"""

import yaml
import cv2
import json
from datetime import datetime
import sys

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

from src import (
    PersonDetector, DeepSORTTracker, PoseEstimator,
    FeatureExtractor, EngagementClassifier, ReportGenerator
)

print("=" * 80)
print("FULL VIDEO ANALYSIS")
print("=" * 80)

# 초기화
print("\n[1/5] 컴포넌트 초기화...")
detector = PersonDetector('models/yolov7.pt', device='cpu')
tracker = DeepSORTTracker()
pose_estimator = PoseEstimator(model_complexity=0)
feature_extractor = FeatureExtractor(config)
classifier = EngagementClassifier(config)
report_generator = ReportGenerator(config)
print("  ✓ 완료")

# 비디오 로드
print("\n[2/5] 비디오 로드...")
video_path = 'data/raw/video_example.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps
print(f"  ✓ {total_frames} 프레임 ({duration_sec:.1f}초)")

# 프레임 처리
print(f"\n[3/5] 프레임 처리 중...")
frame_idx = 0
timeline = []
checkpoint_interval = max(30, total_frames // 10)  # 10% 진행도마다 출력

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame_idx, frame)
        
        pose_results = {}
        for obj in tracked_objects:
            x_min, y_min, x_max, y_max = obj['bbox']
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                pose_results[obj['track_id']] = pose_estimator.estimate(cropped)
        
        timestamp = frame_idx / fps
        frame_features = feature_extractor.extract_per_frame(
            frame_idx, timestamp, tracked_objects, pose_results, fps
        )
        
        for student_id, features in frame_features.items():
            classification = classifier.classify(features)
            timeline.append({
                'timeblock': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'student_id': student_id,
                'features': features,
                'classification': classification,
            })
        
        if (frame_idx + 1) % checkpoint_interval == 0:
            progress = 100 * (frame_idx + 1) / total_frames
            print(f"  {progress:.0f}% ({frame_idx + 1}/{total_frames} 프레임)")
        
        frame_idx += 1
        
    except Exception as e:
        print(f"  ⚠ 프레임 {frame_idx}: {e}")
        frame_idx += 1

cap.release()

# 리포트 생성
print(f"\n[4/5] 리포트 생성 중...")
session_data = {
    'session_name': video_path,
    'session_duration_min': duration_sec / 60,
    'students_tracked': sorted(set(r['student_id'] for r in timeline)),
    'timeline': timeline,
}

report_text = report_generator.generate_report(session_data, output_format='text')
report_html = report_generator.generate_report(session_data, output_format='html')
report_json = report_generator.generate_report(session_data, output_format='json')

# 파일 저장
print(f"\n[5/5] 결과 저장 중...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 텍스트 리포트
report_txt_path = f'outputs/reports/engagement_report_{timestamp}.txt'
with open(report_txt_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"  ✓ {report_txt_path}")

# HTML 리포트
report_html_path = f'outputs/reports/engagement_report_{timestamp}.html'
with open(report_html_path, 'w', encoding='utf-8') as f:
    f.write(report_html)
print(f"  ✓ {report_html_path}")

# JSON 데이터
report_json_path = f'outputs/reports/engagement_data_{timestamp}.json'
with open(report_json_path, 'w', encoding='utf-8') as f:
    f.write(report_json)
print(f"  ✓ {report_json_path}")

# 요약 출력
print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print(f"\n분석 통계:")
print(f"  • 총 프레임: {total_frames}")
print(f"  • 재생 시간: {duration_sec/60:.1f}분")
print(f"  • 감지된 학생: {len(session_data['students_tracked'])}")
print(f"  • 분석 레코드: {len(timeline)}")

# 학생별 참여도 평균
print(f"\n학생별 평균 참여도:")
import pandas as pd
df = pd.DataFrame([
    {
        'student_id': r['student_id'],
        'engagement': r['classification']['engagement_score'],
    }
    for r in timeline
])

for student_id in sorted(df['student_id'].unique()):
    mean_score = df[df['student_id'] == student_id]['engagement'].mean()
    print(f"  {student_id}: {mean_score:.1f}/100")

print(f"\n리포트 저장됨:")
print(f"  TEXT: {report_txt_path}")
print(f"  HTML: {report_html_path}")
print(f"  JSON: {report_json_path}")

EOF

python analyze_video_full.py
```

---

## 4. 시각화 (Visualization)

### 4.1 결과 시각화 스크립트
```bash
cat > visualize_results.py << 'EOF'
"""
분석 결과 시각화
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

# 최신 JSON 리포트 찾기
report_dir = Path('outputs/reports')
json_files = sorted(report_dir.glob('engagement_data_*.json'))

if not json_files:
    print("✗ 리포트 파일이 없습니다. 먼저 analyze_video_full.py를 실행하세요.")
    exit(1)

latest_json = json_files[-1]
print(f"로드 중: {latest_json}")

with open(latest_json) as f:
    session_data = json.load(f)

timeline = session_data['timeline']

# 데이터프레임 생성
records = []
for record in timeline:
    records.append({
        'student_id': record['student_id'],
        'timestamp': record['timestamp'],
        'engagement_score': record['classification']['engagement_score'],
        'engagement_level': record['classification']['engagement_level'],
        'gaze_score': record['classification']['component_scores']['gaze_score'],
        'posture_score': record['classification']['component_scores']['posture_score'],
        'gesture_score': record['classification']['component_scores']['gesture_score'],
        'interaction_score': record['classification']['component_scores']['interaction_score'],
    })

df = pd.DataFrame(records)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Classroom Engagement Analysis', fontsize=16, fontweight='bold')

# 1. 학생별 평균 참여도
ax = axes[0, 0]
student_means = df.groupby('student_id')['engagement_score'].mean().sort_values(ascending=False)
colors = ['green' if x >= 67 else 'orange' if x >= 34 else 'red' for x in student_means.values]
student_means.plot(kind='bar', ax=ax, color=colors)
ax.set_title('Average Engagement by Student')
ax.set_ylabel('Engagement Score (0-100)')
ax.axhline(y=67, color='g', linestyle='--', alpha=0.5, label='Engaged threshold')
ax.axhline(y=33, color='r', linestyle='--', alpha=0.5, label='Disengaged threshold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. 시간별 참여도 추이
ax = axes[0, 1]
for student_id in df['student_id'].unique():
    student_data = df[df['student_id'] == student_id]
    ax.plot(student_data['timestamp'], student_data['engagement_score'], 
            marker='o', label=student_id, alpha=0.7)
ax.set_title('Engagement Over Time')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Engagement Score')
ax.axhline(y=67, color='g', linestyle='--', alpha=0.3)
ax.axhline(y=33, color='r', linestyle='--', alpha=0.3)
ax.legend()
ax.grid(alpha=0.3)

# 3. 컴포넌트 점수 분포
ax = axes[1, 0]
components = ['gaze_score', 'posture_score', 'gesture_score', 'interaction_score']
component_means = [df[col].mean() for col in components]
component_labels = ['Gaze', 'Posture', 'Gesture', 'Interaction']
ax.bar(component_labels, component_means, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
ax.set_title('Average Component Scores')
ax.set_ylabel('Score (0-100)')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

# 4. 참여도 수준 분포 (pie chart)
ax = axes[1, 1]
level_counts = df['engagement_level'].value_counts()
colors_pie = {'engaged': 'green', 'passive': 'orange', 'disengaged': 'red'}
pie_colors = [colors_pie.get(level, 'gray') for level in level_counts.index]
ax.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%', 
       colors=pie_colors, startangle=90)
ax.set_title('Engagement Level Distribution')

plt.tight_layout()
plt.savefig('outputs/visualizations/engagement_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ 시각화 저장됨: outputs/visualizations/engagement_analysis.png")
plt.show()

EOF

python visualize_results.py
```

---

## 5. 트러블슈팅

### 문제: "CUDA out of memory"
```python
# 해결: CPU 사용 또는 경량 모델
config['detection']['device'] = 'cpu'
pose_estimator = PoseEstimator(model_complexity=0)
```

### 문제: "모델을 찾을 수 없음"
```bash
# YOLOv7 자동 다운로드
mkdir -p models
# 또는 수동으로:
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -O models/yolov7.pt
```

### 문제: "비디오에서 사람을 감지하지 못함"
```python
# 1. 신뢰도 임계값 낮추기
config['detection']['confidence_threshold'] = 0.35

# 2. 비디오 해상도 확인
# - 너무 작으면 (< 480p) 감지 어려움
# - OpenCV로 확인: cv2.IMREAD, width/height 확인

# 3. 조명 확인
# - 너무 어두우면 감지율 낮음
# - 카메라 설정 조정
```

---

## 6. 다음 단계

1. ✓ 빠른 테스트로 시스템 동작 확인
2. ✓ 전체 비디오 분석 실행
3. ✓ 결과 리포트 및 시각화 확인
4. → 특징 임계값 조정 (필요시)
5. → 실시간 처리 최적화
6. → 학습 데이터로 ML 모델 훈련
