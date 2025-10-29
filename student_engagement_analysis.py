# -*- coding: utf-8 -*-
"""
STUDENT ENGAGEMENT ANALYSIS - Simplified
학생별 참여도 분석 (개선된 감지 알고리즘)
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

print("=" * 80)
print("STUDENT ENGAGEMENT ANALYSIS - Per-Student Scoring")
print("=" * 80)

# ============================================================================
# STUDENT TRACKER
# ============================================================================

class StudentTracker:
    """학생 ID 추적"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.age_counter = defaultdict(int)
    
    def update(self, detections):
        """detections: [(x1, y1, x2, y2, confidence), ...]"""
        new_tracks = {}
        matched = set()
        
        # 기존 추적과 매칭
        for sid in list(self.tracks.keys()):
            best_dist = 150
            best_idx = -1
            
            x1, y1, x2, y2 = self.tracks[sid]['box']
            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2
            
            for idx, (bx1, by1, bx2, by2, conf) in enumerate(detections):
                if idx in matched:
                    continue
                
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                dist = np.sqrt((tcx - cx)**2 + (tcy - cy)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            if best_idx >= 0:
                x1, y1, x2, y2, conf = detections[best_idx]
                new_tracks[sid] = {'box': (x1, y1, x2, y2)}
                self.age_counter[sid] = 0
                matched.add(best_idx)
            else:
                self.age_counter[sid] += 1
                if self.age_counter[sid] <= 15:
                    new_tracks[sid] = self.tracks[sid]
        
        # 새로운 객체
        for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            if idx not in matched:
                sid = self.next_id
                self.next_id += 1
                new_tracks[sid] = {'box': (x1, y1, x2, y2)}
                self.age_counter[sid] = 0
        
        self.tracks = new_tracks
        
        # 반환
        results = []
        for sid, track in self.tracks.items():
            x1, y1, x2, y2 = track['box']
            results.append((x1, y1, x2, y2, sid))
        
        return results

# ============================================================================
# IMPROVED PERSON DETECTOR
# ============================================================================

class ImprovedDetector:
    """개선된 사람 감지"""
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False
        )
    
    def detect(self, frame):
        """사람 감지 - 여러 방법 조합"""
        detections = []
        
        # 방법 1: 배경 차감
        fgmask = self.bg_subtractor.apply(frame)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 크기 필터
            if area < 300:  # 더 작은 객체 포함
                continue
            if area > 100000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 최소 크기
            if h < 20 or w < 10:
                continue
            
            # 종횡비 확인 (더 관대함)
            aspect = h / max(w, 1)
            if aspect < 1.2:  # 더 낮은 임계값
                continue
            
            # 정규화된 신뢰도
            conf = min(np.sqrt(area) / 200, 1.0)
            
            detections.append((x, y, x + w, y + h, conf))
        
        return detections

# ============================================================================
# ENGAGEMENT ANALYZER
# ============================================================================

class EngagementAnalyzer:
    """참여도 분석"""
    def __init__(self):
        self.prev_frame = None
        self.student_data = defaultdict(lambda: {
            'frames': 0,
            'motion': [],
            'engagement_scores': [],
            'activity': []
        })
    
    def analyze_frame(self, frame, tracked_objects):
        """프레임 분석"""
        results = []
        
        for x1, y1, x2, y2, sid in tracked_objects:
            # 바운드 체크
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            roi = frame[y1:y2, x1:x2]
            
            # 움직임 분석
            motion = np.std(roi)
            
            # 참여도 점수 계산
            if motion > 30:
                engagement = 80
                activity_level = "high"
            elif motion > 15:
                engagement = 65
                activity_level = "medium"
            elif motion > 5:
                engagement = 50
                activity_level = "low"
            else:
                engagement = 35
                activity_level = "minimal"
            
            # 밝기 분석 (얼굴 감지 대체)
            brightness = np.mean(roi)
            if brightness > 150:
                engagement += 10
            
            engagement = min(int(engagement), 100)
            
            # 학생 데이터 업데이트
            self.student_data[sid]['frames'] += 1
            self.student_data[sid]['motion'].append(motion)
            self.student_data[sid]['engagement_scores'].append(engagement)
            self.student_data[sid]['activity'].append(activity_level)
            
            results.append({
                'student_id': sid,
                'engagement': engagement,
                'motion': motion,
                'activity': activity_level,
                'bbox': (x1, y1, x2, y2)
            })
        
        self.prev_frame = frame.copy()
        return results
    
    def get_summary(self):
        """최종 요약"""
        summary = []
        for sid, data in self.student_data.items():
            if data['frames'] == 0:
                continue
            
            avg_engagement = int(np.mean(data['engagement_scores']))
            max_engagement = max(data['engagement_scores'])
            avg_motion = float(np.mean(data['motion']))
            
            summary.append({
                'student_id': int(sid),
                'frames_seen': data['frames'],
                'avg_engagement': avg_engagement,
                'max_engagement': max_engagement,
                'avg_motion': avg_motion,
                'engagement_level': 'high' if avg_engagement >= 70 else (
                    'medium' if avg_engagement >= 50 else 'low'
                )
            })
        
        # 참여도순 정렬
        summary.sort(key=lambda x: x['avg_engagement'], reverse=True)
        return summary

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n[1/4] Loading video...")
video_file = "video_example.mp4"
if not Path(video_file).exists():
    print(f"X Video not found: {video_file}")
    sys.exit(1)

cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"  OK {total_frames} frames at {fps} FPS")

print("\n[2/4] Initializing components...")
detector = ImprovedDetector()
tracker = StudentTracker()
analyzer = EngagementAnalyzer()

print("\n[3/4] Analyzing frames...")
frame_count = 0
max_frames = min(800, total_frames)

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 감지 및 추적
    detections = detector.detect(frame)
    tracked = tracker.update(detections)
    
    # 참여도 분석
    analyzer.analyze_frame(frame, tracked)
    
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"  Frame {frame_count}/{max_frames}")

cap.release()

print("\n[4/4] Generating report...")

# 요약 생성
summary = analyzer.get_summary()

print(f"\n Found {len(summary)} unique students:")
for stats in summary:
    print(f"  Student {stats['student_id']:2d}: "
          f"Engagement={stats['avg_engagement']:3d}/100 ({stats['engagement_level']:6s}), "
          f"Seen={stats['frames_seen']:3d}f, "
          f"Motion={stats['avg_motion']:5.1f}")

# 저장
output_dir = Path('outputs/reports')
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON 저장
json_file = output_dir / f'student_engagement_{timestamp}.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {
            'video': video_file,
            'frames_analyzed': frame_count,
            'fps': fps,
            'timestamp': timestamp,
            'total_students': len(summary)
        },
        'students': summary
    }, f, indent=2, ensure_ascii=False)

print(f"\n  JSON saved: {json_file}")

# HTML 리포트
html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Student Engagement Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        .stats {{ background: white; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }}
        th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f9f9f9; }}
        .high {{ background: #d4edda; color: #155724; font-weight: bold; }}
        .medium {{ background: #fff3cd; color: #856404; font-weight: bold; }}
        .low {{ background: #f8d7da; color: #721c24; font-weight: bold; }}
        .badge {{ padding: 4px 8px; border-radius: 3px; font-size: 0.85em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Classroom Student Engagement Analysis</h1>
        
        <div class="stats">
            <h2>Summary</h2>
            <p><strong>Video:</strong> {video_file}</p>
            <p><strong>Frames Analyzed:</strong> {frame_count}/{total_frames}</p>
            <p><strong>Students Detected:</strong> {len(summary)}</p>
            <p><strong>Analysis Duration:</strong> {frame_count/fps:.1f} seconds</p>
        </div>
        
        <h2>Per-Student Engagement Scores</h2>
        <table>
            <thead>
                <tr>
                    <th>Student ID</th>
                    <th>Engagement Level</th>
                    <th>Avg Score</th>
                    <th>Max Score</th>
                    <th>Frames Seen</th>
                    <th>Motion</th>
                </tr>
            </thead>
            <tbody>
"""

for stats in summary:
    level = stats['engagement_level']
    class_name = level
    level_display = level.upper()
    
    html += f"""
                <tr>
                    <td><strong>Student {stats['student_id']}</strong></td>
                    <td><span class="badge {class_name}">{level_display}</span></td>
                    <td>{stats['avg_engagement']}/100</td>
                    <td>{stats['max_engagement']}/100</td>
                    <td>{stats['frames_seen']}</td>
                    <td>{stats['avg_motion']:.1f}</td>
                </tr>
"""

# 통계
high = len([s for s in summary if s['engagement_level'] == 'high'])
medium = len([s for s in summary if s['engagement_level'] == 'medium'])
low = len([s for s in summary if s['engagement_level'] == 'low'])

html += f"""
            </tbody>
        </table>
        
        <div class="stats">
            <h2>Engagement Distribution</h2>
            <p><strong>High (70+):</strong> {high} students</p>
            <p><strong>Medium (50-69):</strong> {medium} students</p>
            <p><strong>Low (<50):</strong> {low} students</p>
            <p><strong>Average Class Engagement:</strong> {int(np.mean([s['avg_engagement'] for s in summary]))}/100</p>
        </div>
        
        <p style="color: #999; font-size: 0.9em;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>
"""

html_file = output_dir / f'student_engagement_{timestamp}.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"  HTML saved: {html_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
