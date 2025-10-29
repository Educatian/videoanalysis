# -*- coding: utf-8 -*-
"""
Quick test: Analyze first 30 frames
"""

import yaml
import cv2
import numpy as np
import sys
from pathlib import Path

print("=" * 80)
print("QUICK VIDEO TEST - First 30 frames analysis")
print("=" * 80)

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Import modules
from src import (
    PersonDetector, DeepSORTTracker, PoseEstimator,
    FeatureExtractor, EngagementClassifier
)

# 1. Initialize components
print("\n[1/4] 컴포넌트 초기화 중...")
try:
    detector = PersonDetector(
        model_path='models/yolov7.pt',
        confidence_threshold=0.45,
        device='cpu',
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
        model_complexity=0,
        enable_hand_tracking=True,
    )
    print("  ✓ MediaPipe Pose 준비됨")
except Exception as e:
    print(f"  ✗ Pose estimator 초기화 실패: {e}")
    sys.exit(1)

feature_extractor = FeatureExtractor(config)
classifier = EngagementClassifier(config)
print("  ✓ Feature Extractor & Classifier 준비됨")

# 2. Load video
print("\n[2/4] 비디오 로드 중...")
video_path = 'video_example.mp4'
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

# 3. Process frames
print("\n[3/4] 프레임 처리 중 (최대 30프레임)...")
frame_idx = 0
max_frames = min(30, total_frames)
results = []

while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Detection
        detections = detector.detect(frame)
        
        # Tracking
        tracked_objects = tracker.update(detections, frame_idx, frame)
        
        # Pose estimation
        pose_results = {}
        for obj in tracked_objects:
            track_id = obj['track_id']
            x_min, y_min, x_max, y_max = obj['bbox']
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                pose_results[track_id] = pose_estimator.estimate(cropped)
        
        # Feature extraction
        timestamp = frame_idx / fps
        frame_features = feature_extractor.extract_per_frame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            tracked_objects=tracked_objects,
            pose_results=pose_results,
            fps=fps,
        )
        
        # Classification
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

# 4. Analyze results
print("\n[4/4] 결과 분석 중...")
print("\n" + "=" * 80)
print("분석 결과")
print("=" * 80)

if not results:
    print("결과 없음 - 비디오에 인식된 사람이 없을 수 있습니다")
    sys.exit(0)

# Statistics by student
all_students = set()
for result in results:
    for student_id in result['classifications'].keys():
        all_students.add(student_id)

print(f"\n감지된 학생: {len(all_students)}명")
print(f"처리된 프레임: {len(results)}")

if all_students:
    for student_id in sorted(all_students):
        print(f"\n【{student_id}】")
        scores = []
        
        for result in results:
            if student_id in result['classifications']:
                score = result['classifications'][student_id]['engagement_score']
                level = result['classifications'][student_id]['engagement_level']
                scores.append(score)
                
                # Show first and last frame details
                if result['frame_idx'] == 0 or result['frame_idx'] == len(results) - 1:
                    print(f"  [Frame {result['frame_idx']}] 참여도: {score:.1f}/100 ({level})")
                    
                    features = result['classifications'][student_id]['feature_values']
                    gaze = features.get('gaze_mean_deg', 0)
                    posture = features.get('posture_stability_std_deg', 0)
                    gesture = features.get('gesture_frequency_permin', 0)
                    
                    print(f"    • 시선: {gaze:.1f}° | 자세: ±{posture:.1f}° | 손: {gesture:.1f}/분")
        
        if scores:
            print(f"  평균 참여도: {np.mean(scores):.1f}/100 (±{np.std(scores):.1f})")

print("\n" + "=" * 80)
print("✓ 테스트 완료!")
print("=" * 80)
print("\n전체 비디오 분석을 원하시면:")
print("  python analyze_video_full.py")
print("\n결과 시각화:")
print("  python visualize_results.py")
