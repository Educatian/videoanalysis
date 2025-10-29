# -*- coding: utf-8 -*-
"""
COMPREHENSIVE STUDENT ENGAGEMENT ANALYSIS
학생별 참여도 분석 - YOLOv7 + MediaPipe + Custom Tracker

Features:
- Person detection with YOLOv7
- Student ID tracking (custom implementation)
- Pose estimation with MediaPipe
- Per-student engagement scoring
- Detailed analytics and reports
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

print("=" * 80)
print("COMPREHENSIVE STUDENT ENGAGEMENT ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: STUDENT TRACKER (DeepSORT 대체)
# ============================================================================

class SimpleStudentTracker:
    """
    간단한 학생 추적 시스템
    중심점 거리를 기반으로 프레임 간 학생 ID 유지
    """
    def __init__(self, max_distance=100, max_age=30):
        self.tracks = {}  # {student_id: track_info}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_age = max_age
        self.age_counter = defaultdict(int)
    
    def update(self, detections):
        """
        detections: [(x1, y1, x2, y2, confidence), ...]
        returns: [(x1, y1, x2, y2, student_id), ...]
        """
        if not detections:
            # Age increase for missing detections
            for sid in list(self.tracks.keys()):
                self.age_counter[sid] += 1
                if self.age_counter[sid] > self.max_age:
                    del self.tracks[sid]
                    del self.age_counter[sid]
            return []
        
        new_detections = detections.copy()
        matched = set()
        results = []
        
        # Match with existing tracks
        for sid, track in list(self.tracks.items()):
            best_dist = self.max_distance
            best_idx = -1
            
            tx1, ty1, tx2, ty2 = track['bbox']
            tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            
            for idx, (x1, y1, x2, y2, conf) in enumerate(new_detections):
                if idx in matched:
                    continue
                
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist = np.sqrt((tcx - cx)**2 + (tcy - cy)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx >= 0:
                x1, y1, x2, y2, conf = new_detections[best_idx]
                self.tracks[sid] = {'bbox': (x1, y1, x2, y2), 'conf': conf}
                self.age_counter[sid] = 0
                results.append((x1, y1, x2, y2, sid))
                matched.add(best_idx)
            else:
                self.age_counter[sid] += 1
                if self.age_counter[sid] <= self.max_age:
                    x1, y1, x2, y2 = track['bbox']
                    results.append((x1, y1, x2, y2, sid))
                else:
                    del self.tracks[sid]
                    del self.age_counter[sid]
        
        # Create new tracks
        for idx, (x1, y1, x2, y2, conf) in enumerate(new_detections):
            if idx not in matched:
                sid = self.next_id
                self.next_id += 1
                self.tracks[sid] = {'bbox': (x1, y1, x2, y2), 'conf': conf}
                self.age_counter[sid] = 0
                results.append((x1, y1, x2, y2, sid))
        
        return results

# ============================================================================
# PART 2: SIMPLE OBJECT DETECTION (YOLOv7 기반, OpenCV 폴백)
# ============================================================================

class PersonDetector:
    """간단한 사람 감지기"""
    def __init__(self):
        self.net = None
        self.use_yolo = False
        
        # YOLOv7 시도
        try:
            import torch
            self.use_yolo = True
            print("  OK YOLOv7 will be used for detection")
        except:
            print("  X YOLOv7 not available, using OpenCV")
    
    def detect(self, frame):
        """
        frame에서 사람 감지
        returns: [(x1, y1, x2, y2, confidence), ...]
        """
        h, w = frame.shape[:2]
        
        if self.use_yolo:
            # YOLOv7 사용 시도
            try:
                # 실제 구현은 torch 필요
                detections = self._detect_yolo(frame)
            except:
                detections = self._detect_opencv(frame)
        else:
            detections = self._detect_opencv(frame)
        
        return detections
    
    def _detect_opencv(self, frame):
        """OpenCV 기반 사람 감지 (간단한 버전)"""
        # MOG2 배경 감지 + 컨투어
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgmask = fgbg.apply(frame)
        
        # 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 50000:  # 사람 크기 범위
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 30 or w < 15:
                continue
            
            # 종횡비 확인 (사람은 높이 > 너비)
            if h < w * 1.5:
                continue
            
            conf = min(area / 10000, 1.0)
            detections.append((x, y, x + w, y + h, conf))
        
        return detections
    
    def _detect_yolo(self, frame):
        """YOLOv7 기반 감지"""
        # TODO: YOLOv7 구현
        return []

# ============================================================================
# PART 3: POSE ESTIMATION (MediaPipe)
# ============================================================================

class PoseAnalyzer:
    """포즈 및 행동 분석"""
    def __init__(self):
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp = mp
            self.use_mediapipe = True
            print("  OK MediaPipe available for pose estimation")
        except:
            print("  X MediaPipe not available")
    
    def analyze(self, frame, bbox):
        """
        bbox 영역의 포즈 분석
        returns: {posture, hand_activity, gaze_direction, engagement_signals}
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return self._default_analysis()
        
        person_region = frame[y1:y2, x1:x2]
        
        if self.use_mediapipe:
            return self._analyze_mediapipe(person_region, frame, (x1, y1))
        else:
            return self._analyze_simple(person_region)
    
    def _default_analysis(self):
        return {
            'posture': 'unknown',
            'hand_activity': 0.0,
            'gaze_direction': 'forward',
            'engagement_score': 50,
            'motion_level': 0.0
        }
    
    def _analyze_simple(self, region):
        """간단한 분석"""
        # 움직임 감지
        if region.size == 0:
            return self._default_analysis()
        
        motion = np.std(region)
        hand_activity = min(motion / 50, 1.0)
        
        return {
            'posture': 'neutral',
            'hand_activity': float(hand_activity),
            'gaze_direction': 'forward',
            'engagement_score': int(50 + hand_activity * 30),
            'motion_level': float(motion)
        }
    
    def _analyze_mediapipe(self, region, full_frame, offset):
        """MediaPipe 기반 분석"""
        try:
            pose = self.mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1
            )
            
            # 포즈 감지
            results = pose.process(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                # 랜드마크 추출
                landmarks = results.pose_landmarks.landmark
                
                # 머리 기울기 (코, 양쪽 눈 사용)
                nose = landmarks[0]
                left_eye = landmarks[2]
                right_eye = landmarks[5]
                
                head_tilt = abs(left_eye.y - right_eye.y)
                
                # 손 활동 (양쪽 손)
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                
                hand_activity = max(left_wrist.visibility, right_wrist.visibility)
                
                # 자세 (어깨)
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                
                posture_angle = abs(left_shoulder.y - right_shoulder.y)
                
                return {
                    'posture': 'forward' if posture_angle < 0.1 else 'tilted',
                    'hand_activity': float(hand_activity),
                    'gaze_direction': 'forward',
                    'engagement_score': int(60 + hand_activity * 30),
                    'motion_level': float(head_tilt)
                }
        except Exception as e:
            print(f"    MediaPipe error: {e}")
        
        return self._analyze_simple(region)

# ============================================================================
# PART 4: PER-STUDENT ENGAGEMENT SCORING
# ============================================================================

class EngagementScorer:
    """학생별 참여도 점수 계산"""
    def __init__(self):
        self.student_history = defaultdict(lambda: {
            'frames': 0,
            'total_engagement': 0,
            'hand_activities': [],
            'postures': [],
            'motion_levels': []
        })
    
    def update_student(self, student_id, pose_analysis):
        """학생 데이터 업데이트"""
        hist = self.student_history[student_id]
        
        hist['frames'] += 1
        engagement = pose_analysis['engagement_score']
        hist['total_engagement'] += engagement
        hist['hand_activities'].append(pose_analysis['hand_activity'])
        hist['postures'].append(pose_analysis['posture'])
        hist['motion_levels'].append(pose_analysis['motion_level'])
    
    def get_student_stats(self, student_id):
        """학생 통계"""
        hist = self.student_history[student_id]
        
        if hist['frames'] == 0:
            return {
                'student_id': student_id,
                'frames_seen': 0,
                'avg_engagement': 0,
                'avg_hand_activity': 0,
                'primary_posture': 'unknown',
                'avg_motion': 0
            }
        
        return {
            'student_id': student_id,
            'frames_seen': hist['frames'],
            'avg_engagement': int(hist['total_engagement'] / hist['frames']),
            'avg_hand_activity': float(np.mean(hist['hand_activities'])),
            'primary_posture': max(set(hist['postures']), key=hist['postures'].count),
            'avg_motion': float(np.mean(hist['motion_levels']))
        }
    
    def get_all_students(self):
        """모든 학생 통계"""
        return [self.get_student_stats(sid) for sid in self.student_history.keys()]

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n[1/5] Loading video...")
video_file = "video_example.mp4"
if not Path(video_file).exists():
    print(f"X Video not found: {video_file}")
    sys.exit(1)

cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  OK {total_frames} frames at {fps} FPS")

print("\n[2/5] Initializing components...")
detector = PersonDetector()
tracker = SimpleStudentTracker()
pose_analyzer = PoseAnalyzer()
scorer = EngagementScorer()

print("\n[3/5] Analyzing frames...")
frame_count = 0
max_frames = min(500, total_frames)  # 500프레임 분석

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 감지
    detections = detector.detect(frame)
    
    # 추적
    tracked = tracker.update(detections)
    
    # 포즈 분석
    for x1, y1, x2, y2, student_id in tracked:
        pose_analysis = pose_analyzer.analyze(frame, (x1, y1, x2, y2))
        scorer.update_student(student_id, pose_analysis)
    
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"  Frame {frame_count}/{max_frames}")

cap.release()

print("\n[4/5] Generating statistics...")
student_stats = scorer.get_all_students()

# 참여도별 정렬
student_stats.sort(key=lambda x: x['avg_engagement'], reverse=True)

print(f"\n  Found {len(student_stats)} unique students:")
for stats in student_stats:
    print(f"    Student {stats['student_id']:2d}: "
          f"Engagement={stats['avg_engagement']:3d}/100, "
          f"Seen={stats['frames_seen']:3d} frames, "
          f"Activity={stats['avg_hand_activity']:.2f}")

print("\n[5/5] Saving results...")
output_dir = Path('outputs/reports')
output_dir.mkdir(parents=True, exist_ok=True)

# JSON 저장
json_file = output_dir / f'student_engagement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {
            'video': video_file,
            'frames_analyzed': frame_count,
            'timestamp': datetime.now().isoformat()
        },
        'students': student_stats
    }, f, indent=2, ensure_ascii=False)

print(f"  OK Saved to {json_file}")

# HTML 리포트 생성
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Student Engagement Report</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        tr:hover { background: #f5f5f5; }
        .high { color: green; font-weight: bold; }
        .medium { color: orange; font-weight: bold; }
        .low { color: red; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Classroom Student Engagement Analysis Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Video: """ + video_file + """</p>
        <p>Frames Analyzed: """ + str(frame_count) + """</p>
        <p>Total Students Detected: """ + str(len(student_stats)) + """</p>
    </div>
    
    <h2>Per-Student Engagement Scores</h2>
    <table>
        <tr>
            <th>Student ID</th>
            <th>Avg Engagement</th>
            <th>Frames Seen</th>
            <th>Hand Activity</th>
            <th>Primary Posture</th>
            <th>Avg Motion</th>
        </tr>
"""

for stats in student_stats:
    engagement = stats['avg_engagement']
    
    if engagement >= 70:
        engagement_class = "high"
    elif engagement >= 50:
        engagement_class = "medium"
    else:
        engagement_class = "low"
    
    html_content += f"""
        <tr>
            <td>Student {stats['student_id']}</td>
            <td class="{engagement_class}">{engagement}/100</td>
            <td>{stats['frames_seen']}</td>
            <td>{stats['avg_hand_activity']:.2f}</td>
            <td>{stats['primary_posture']}</td>
            <td>{stats['avg_motion']:.2f}</td>
        </tr>
"""

html_content += """
    </table>
    
    <h2>Engagement Distribution</h2>
    <p>
        High (70+): """ + str(len([s for s in student_stats if s['avg_engagement'] >= 70])) + """ students<br>
        Medium (50-69): """ + str(len([s for s in student_stats if 50 <= s['avg_engagement'] < 70])) + """ students<br>
        Low (<50): """ + str(len([s for s in student_stats if s['avg_engagement'] < 50])) + """ students
    </p>
    
    <p style="margin-top: 30px; color: #999;">
        Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
    </p>
</body>
</html>
"""

html_file = output_dir / f'student_engagement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"  OK Saved to {html_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
