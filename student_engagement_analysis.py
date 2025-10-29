# -*- coding: utf-8 -*-
"""
===============================================================================
STUDENT ENGAGEMENT ANALYSIS - 학생별 참여도 분석 시스템
===============================================================================

📚 튜토리얼: 컴퓨터 비전을 이용한 학생 참여도 자동 분석

이 스크립트는 다음 4단계로 작동합니다:

1️⃣ STEP 1: 객체 감지 (Object Detection)
   - 비디오에서 학생(사람)을 감지합니다
   - 배경 차감 + 모폴로지 연산 사용
   - 결과: 각 프레임에서 발견된 사람의 위치 좌표

2️⃣ STEP 2: 다중 객체 추적 (Multi-Object Tracking)
   - 같은 학생이 프레임마다 같은 ID를 유지하도록 추적합니다
   - 중심점 거리 기반 매칭 알고리즘 사용
   - 결과: 각 학생에게 고유한 ID 할당

3️⃣ STEP 3: 참여도 분석 (Engagement Analysis)
   - 움직임, 밝기 등을 분석하여 참여도 점수 계산
   - 0~100 범위의 점수 생성
   - 결과: High/Medium/Low 분류

4️⃣ STEP 4: 리포트 생성 (Report Generation)
   - JSON과 HTML 형식의 리포트 생성
   - 학생별 상세 통계 포함
   - 결과: 분석 결과 저장

===============================================================================
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
# PART 1: STUDENT TRACKER (학생 추적 시스템)
# ============================================================================
# 
# 목적: 프레임 간에 같은 학생이 같은 ID를 유지하도록 추적
# 알고리즘: 중심점 거리 기반 매칭 (Centroid Matching)
#
# 원리:
# - 이전 프레임의 학생 위치와 현재 프레임의 감지된 사람을 비교
# - 가장 가까운 위치에 있는 것을 같은 학생으로 간주
# - 추적 실패 시 일시적으로 유지 (오클루전 처리)
#

class StudentTracker:
    """학생 ID 추적 - DeepSORT 대체 구현"""
    
    def __init__(self):
        """추적기 초기화"""
        self.tracks = {}              # 현재 추적 중인 학생들 {student_id: {'box': (x1, y1, x2, y2)}}
        self.next_id = 1              # 다음 할당할 학생 ID
        self.age_counter = defaultdict(int)  # 각 학생의 나이 (추적 실패 프레임 수)
    
    def update(self, detections):
        """
        새 프레임에서 감지된 사람들과 기존 추적을 매칭
        
        Args:
            detections: 감지된 사람들 [(x1, y1, x2, y2, confidence), ...]
        
        Returns:
            추적된 학생들 [(x1, y1, x2, y2, student_id), ...]
        """
        new_tracks = {}
        matched = set()
        
        # ========== 단계 1: 기존 추적과 새 감지 매칭 ==========
        # 이전 프레임에서 추적하던 각 학생에 대해
        for sid in list(self.tracks.keys()):
            best_dist = 250  # 최대 추적 거리 (픽셀 단위)
            best_idx = -1    # 가장 가까운 감지 인덱스
            
            # 이전 프레임에서 학생의 위치
            x1, y1, x2, y2 = self.tracks[sid]['box']
            # 중심점 계산
            tcx, tcy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 현재 감지된 모든 사람 중에서 가장 가까운 것 찾기
            for idx, (bx1, by1, bx2, by2, conf) in enumerate(detections):
                if idx in matched:  # 이미 매칭된 것 건너뛰기
                    continue
                
                # 현재 감지의 중심점
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                # 유클리드 거리 계산
                dist = np.sqrt((tcx - cx)**2 + (tcy - cy)**2)
                
                # 가장 가까운 것 기록
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            
            # ========== 단계 1-1: 매칭 성공 (같은 학생으로 간주) ==========
            if best_idx >= 0:
                x1, y1, x2, y2, conf = detections[best_idx]
                new_tracks[sid] = {'box': (x1, y1, x2, y2)}
                self.age_counter[sid] = 0  # 추적 성공했으므로 나이 리셋
                matched.add(best_idx)      # 이 감지는 매칭됨 표시
            
            # ========== 단계 1-2: 매칭 실패 (오클루전 처리) ==========
            else:
                self.age_counter[sid] += 1  # 이 학생이 안 보인 프레임 수 증가
                # 최대 30프레임까지는 같은 학생으로 유지 (일시적 가림 대비)
                if self.age_counter[sid] <= 30:
                    new_tracks[sid] = self.tracks[sid]
        
        # ========== 단계 2: 새로운 학생에 ID 할당 ==========
        # 아직 매칭되지 않은 감지들 (새로운 학생)
        for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            if idx not in matched:
                # 새로운 학생 ID 할당
                sid = self.next_id
                self.next_id += 1
                new_tracks[sid] = {'box': (x1, y1, x2, y2)}
                self.age_counter[sid] = 0
        
        self.tracks = new_tracks
        
        # ========== 단계 3: 결과 반환 ==========
        # 추적된 모든 학생들 [(x1, y1, x2, y2, student_id), ...]
        results = []
        for sid, track in self.tracks.items():
            x1, y1, x2, y2 = track['box']
            results.append((x1, y1, x2, y2, sid))
        
        return results

# ============================================================================
# PART 2: IMPROVED PERSON DETECTOR (개선된 사람 감지기)
# ============================================================================
#
# 목적: 비디오에서 사람(학생)을 감지합니다
# 방법: OpenCV의 배경 차감 + 모폴로지 연산
#
# 원리:
# 1. MOG2 배경 감지: 배경과 움직이는 객체를 분리
# 2. 모폴로지 연산: 노이즈 제거 및 객체 강화
# 3. 컨투어 추출: 객체의 경계선 추출
# 4. 필터링: 사람 크기와 형태에 맞는 것만 선택
#

class ImprovedDetector:
    """개선된 사람 감지기"""
    
    def __init__(self):
        """
        감지기 초기화
        
        MOG2 (Mixture of Gaussians): 
        - 배경 모델을 여러 개의 가우시안 분포로 표현
        - 변화하는 조명과 동적 배경에 강함
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False  # 그림자 감지 비활성화 (오감지 방지)
        )
    
    def detect(self, frame):
        """
        프레임에서 사람을 감지합니다
        
        Args:
            frame: 입력 비디오 프레임 (H x W x 3)
        
        Returns:
            감지된 사람들 [(x1, y1, x2, y2, confidence), ...]
        """
        detections = []
        
        # ========== 단계 1: 배경 차감 ==========
        # 각 픽셀을 배경/전경으로 분류
        fgmask = self.bg_subtractor.apply(frame)
        # fgmask: 0 (배경) 또는 255 (전경)
        
        # ========== 단계 2: 모폴로지 연산 (노이즈 제거) ==========
        # 커널: 7x7 타원 모양 구조화 요소
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # MORPH_CLOSE: 작은 구멍 채우기 (객체 내부의 검은 점 제거)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        
        # MORPH_OPEN: 작은 노이즈 제거 (객체 주변의 흰 점 제거)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # ========== 단계 3: 컨투어 추출 ==========
        # 전경 픽셀들의 경계선 찾기
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ========== 단계 4: 각 컨투어를 필터링 ==========
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # ========== 단계 4-1: 크기 필터 ==========
            # 너무 작은 노이즈 제거 (area < 5000 픽셀)
            if area < 5000:
                continue
            # 너무 큰 배경 제거 (area > 50000 픽셀)
            if area > 50000:
                continue
            
            # 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(cnt)
            
            # ========== 단계 4-2: 최소 크기 필터 ==========
            # 너무 작은 객체 제외
            if h < 50 or w < 20:  # 높이 최소 50픽셀, 너비 최소 20픽셀
                continue
            
            # ========== 단계 4-3: 종횡비(Aspect Ratio) 필터 ==========
            # 사람은 일반적으로 세로가 가로보다 훨씬 길다
            aspect = h / max(w, 1)  # aspect = height / width
            if aspect < 1.8:  # 너무 방광한 모양 제외 (사람이 아님)
                continue
            
            # ========== 단계 4-4: 신뢰도 계산 ==========
            # 면적이 클수록 더 확실한 감지
            conf = min(np.sqrt(area) / 200, 1.0)  # 0~1 범위
            
            # ========== 단계 4-5: 감지 결과 저장 ==========
            detections.append((x, y, x + w, y + h, conf))
        
        return detections

# ============================================================================
# PART 3: ENGAGEMENT ANALYZER (참여도 분석기)
# ============================================================================
#
# 목적: 각 학생의 참여도를 계산합니다
# 지표: 움직임 (Motion) + 밝기 (Brightness)
#
# 알고리즘:
# - 움직임이 많을수록 참여도 높음
# - 얼굴 밝기가 높을수록 참여도 높음
#

class EngagementAnalyzer:
    """참여도 분석기"""
    
    def __init__(self):
        """분석기 초기화"""
        self.prev_frame = None
        # 각 학생의 데이터 누적
        self.student_data = defaultdict(lambda: {
            'frames': 0,              # 본 프레임 수
            'motion': [],             # 움직임 값들
            'engagement_scores': [],  # 참여도 점수들
            'activity': []            # 활동 수준들
        })
    
    def analyze_frame(self, frame, tracked_objects):
        """
        프레임의 각 학생 참여도 분석
        
        Args:
            frame: 현재 프레임
            tracked_objects: 추적된 학생들 [(x1, y1, x2, y2, student_id), ...]
        
        Returns:
            분석 결과들
        """
        results = []
        
        # 각 추적된 학생에 대해
        for x1, y1, x2, y2, student_id in tracked_objects:
            # ========== 단계 1: 바운딩 박스 검증 ==========
            # 이미지 범위 내로 조정
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # 유효한 영역 확인
            if x2 <= x1 or y2 <= y1:
                continue
            
            # ========== 단계 2: 학생 영역 추출 (ROI) ==========
            # Region of Interest
            roi = frame[y1:y2, x1:x2]
            
            # ========== 단계 3: 움직임 분석 ==========
            # 표준편차가 클수록 픽셀 변화(움직임)가 많음
            motion = np.std(roi)
            
            # ========== 단계 4: 참여도 점수 계산 ==========
            # 움직임 레벨에 따른 기본 점수 할당
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
            
            # ========== 단계 5: 밝기 보너스 ==========
            # 얼굴/피부가 보이는 밝은 영역이 있을 확률이 높음
            brightness = np.mean(roi)
            if brightness > 150:  # 밝으면 얼굴이 보인다고 가정
                engagement += 10
            
            # ========== 단계 6: 점수 정규화 ==========
            engagement = min(int(engagement), 100)
            
            # ========== 단계 7: 학생 데이터 업데이트 ==========
            self.student_data[student_id]['frames'] += 1
            self.student_data[student_id]['motion'].append(motion)
            self.student_data[student_id]['engagement_scores'].append(engagement)
            self.student_data[student_id]['activity'].append(activity_level)
            
            results.append({
                'student_id': student_id,
                'engagement': engagement,
                'motion': motion,
                'activity': activity_level,
                'bbox': (x1, y1, x2, y2)
            })
        
        self.prev_frame = frame.copy()
        return results
    
    def get_summary(self):
        """
        모든 학생의 최종 통계 생성
        
        Returns:
            학생별 통계 리스트
        """
        summary = []
        
        # 각 학생에 대해
        for student_id, data in self.student_data.items():
            if data['frames'] == 0:
                continue
            
            # ========== 단계 1: 통계 계산 ==========
            avg_engagement = int(np.mean(data['engagement_scores']))
            max_engagement = max(data['engagement_scores'])
            avg_motion = float(np.mean(data['motion']))
            
            # ========== 단계 2: 참여도 수준 분류 ==========
            if avg_engagement >= 70:
                engagement_level = 'high'
            elif avg_engagement >= 50:
                engagement_level = 'medium'
            else:
                engagement_level = 'low'
            
            # ========== 단계 3: 결과 저장 ==========
            summary.append({
                'student_id': int(student_id),
                'frames_seen': data['frames'],
                'avg_engagement': avg_engagement,
                'max_engagement': max_engagement,
                'avg_motion': avg_motion,
                'engagement_level': engagement_level
            })
        
        # 참여도순으로 정렬 (높은 것부터)
        summary.sort(key=lambda x: x['avg_engagement'], reverse=True)
        return summary

# ============================================================================
# MAIN ANALYSIS - 메인 분석 루프
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

# ========== 메인 루프: 각 프레임 처리 ==========
while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1️⃣ 감지 (Step 1: Detection)
    detections = detector.detect(frame)
    
    # 2️⃣ 추적 (Step 2: Tracking)
    tracked = tracker.update(detections)
    
    # 3️⃣ 분석 (Step 3: Analysis)
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
