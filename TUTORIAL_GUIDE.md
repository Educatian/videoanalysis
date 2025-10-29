# 📚 학생 참여도 분석 시스템 - 완전 튜토리얼 가이드

## 🎯 시스템 개요

이 시스템은 **컴퓨터 비전**을 사용하여 교실의 학생 참여도를 자동으로 분석합니다.

```
비디오 입력
    ↓
[STEP 1] 객체 감지 (Object Detection)
    ↓
[STEP 2] 다중 객체 추적 (Multi-Object Tracking)
    ↓
[STEP 3] 참여도 분석 (Engagement Analysis)
    ↓
[STEP 4] 리포트 생성 (Report Generation)
    ↓
JSON + HTML 리포트
```

---

## 📖 상세 설명

### STEP 1️⃣: 객체 감지 (Object Detection)

#### 목표
비디오의 각 프레임에서 **학생(사람)의 위치**를 찾습니다.

#### 방법: MOG2 배경 차감

```python
# 단계 1: 배경 모델 학습 및 적용
fgmask = cv2.createBackgroundSubtractorMOG2()
fgmask = fgmask.apply(frame)

# 결과: 배경 = 0 (검은색), 전경 = 255 (흰색)
```

**MOG2란?**
- Mixture of Gaussians (가우시안 혼합 모델)
- 배경을 여러 확률 분포로 모델링
- 조명 변화와 동적 배경에 강함

#### 단계별 처리

```
1. 원본 프레임 입력
   [칼라 이미지]

2. MOG2 배경 차감
   [배경은 검은색, 움직이는 것만 흰색]

3. 모폴로지 연산 (노이즈 제거)
   - MORPH_CLOSE: 구멍 채우기
   - MORPH_OPEN: 노이즈 제거
   [깔끔한 흰색 영역]

4. 컨투어 추출
   [각 영역의 경계선]

5. 필터링 (사람만 선택)
   - 크기: 5000 < area < 50000 픽셀
   - 높이: h > 50 픽셀
   - 너비: w > 20 픽셀
   - 종횡비: height/width > 1.8
   [최종 감지된 사람들]
```

#### 코드

```python
class ImprovedDetector:
    def detect(self, frame):
        # MOG2 배경 차감
        fgmask = self.bg_subtractor.apply(frame)
        
        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        # 컨투어 추출
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 필터링
            if area < 5000 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = h / max(w, 1)
            
            if aspect < 1.8:
                continue
            
            # 신뢰도 계산
            conf = min(np.sqrt(area) / 200, 1.0)
            detections.append((x, y, x + w, y + h, conf))
        
        return detections
```

#### 결과
- **입력**: 비디오 프레임
- **출력**: [(x1, y1, x2, y2, confidence), ...]
- **예**: [(100, 50, 250, 400, 0.95), (500, 100, 650, 450, 0.92)]

---

### STEP 2️⃣: 다중 객체 추적 (Multi-Object Tracking)

#### 목표
같은 학생이 프레임마다 **같은 ID를 유지**하도록 추적합니다.

#### 문제
- 프레임 1: 학생이 위치 (100, 50)에 감지 → Student 1
- 프레임 2: 같은 학생이 위치 (105, 55)에 감지 → ??? (같은 학생인가 새로운 학생인가?)

#### 해결: 중심점 거리 기반 매칭

```python
# 이전 프레임의 학생
prev_center = (100, 50)  # 중심점

# 현재 프레임의 감지들
current_detections = [
    (95, 45),    # 거리 = 7.1 ← 가장 가까움!
    (200, 200),  # 거리 = 212.1
]

# 가장 가까운 것 = 같은 학생
best_match = min(current_detections, key=lambda x: distance(prev_center, x))
```

#### 알고리즘

```
프레임 N-1 (이전):
- Student 1: 중심 (100, 50)
- Student 2: 중심 (300, 100)

프레임 N (현재 감지):
- 감지 A: 중심 (105, 55)  ← Student 1과 거리 7
- 감지 B: 중심 (305, 105) ← Student 2와 거리 7
- 감지 C: 중심 (500, 200) ← 새로운 학생

매칭 결과:
- 감지 A → Student 1 (매칭 성공)
- 감지 B → Student 2 (매칭 성공)
- 감지 C → Student 3 (새 ID 할당)
```

#### 오클루션 처리 (Occlusion Handling)

경우에 따라 학생이 일시적으로 보이지 않을 수 있습니다:
- 다른 학생이 가림
- 화면 밖으로 나감
- 급격한 움직임

**해결**: 최대 30프레임까지 같은 ID 유지

```python
if best_match_found:
    # 매칭 성공
    student_age[student_id] = 0  # 리셋
else:
    # 매칭 실패
    student_age[student_id] += 1
    
    if student_age[student_id] <= 30:
        # 아직 같은 학생으로 간주
        continue_tracking(student_id)
    else:
        # 30프레임 이상 안 보임 → 다른 학생
        delete_track(student_id)
```

#### 코드

```python
class StudentTracker:
    def update(self, detections):
        # 이전 학생들과 현재 감지 매칭
        for prev_student_id, prev_center in self.tracks.items():
            best_distance = 250  # 최대 거리 임계값
            best_match = -1
            
            # 현재 감지 중 가장 가까운 것 찾기
            for idx, current_detection in enumerate(detections):
                dist = euclidean_distance(prev_center, current_detection)
                
                if dist < best_distance:
                    best_distance = dist
                    best_match = idx
            
            if best_match >= 0:
                # 매칭 성공
                self.tracks[prev_student_id] = detections[best_match]
                self.age[prev_student_id] = 0
                detections[best_match] = 'matched'  # 표시
            else:
                # 매칭 실패
                self.age[prev_student_id] += 1
                
                if self.age[prev_student_id] > 30:
                    delete_track(prev_student_id)
        
        # 매칭되지 않은 감지 = 새로운 학생
        for detection in detections:
            if not matched(detection):
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = detection
```

#### 결과
- **입력**: [(x1, y1, x2, y2, confidence), ...]
- **출력**: [(x1, y1, x2, y2, student_id), ...]
- **예**: [(100, 50, 250, 400, 1), (500, 100, 650, 450, 2)]

---

### STEP 3️⃣: 참여도 분석 (Engagement Analysis)

#### 목표
각 학생의 **참여도 점수**를 계산합니다 (0-100).

#### 지표

**움직임 (Motion)**
```python
# ROI 내의 픽셀 변화 정도
motion = np.std(roi)

# 해석:
# motion > 30  → 많이 움직임 → 높은 참여도
# 15 < motion < 30 → 적당히 움직임 → 중간 참여도
# motion < 15 → 거의 안 움직임 → 낮은 참여도
```

**밝기 (Brightness)**
```python
# 얼굴이 보이는 밝은 영역 비율
brightness = np.mean(roi)

# 해석:
# brightness > 150 → 얼굴이 보임 → 참여 신호
# brightness < 150 → 어두움 → 참여 감소
```

#### 점수 계산

```
기본 점수:
  if motion > 30:        score = 80
  elif motion > 15:      score = 65
  elif motion > 5:       score = 50
  else:                  score = 35

밝기 보너스:
  if brightness > 150:   score += 10

최종 점수:
  score = min(score, 100)

분류:
  if score >= 70:        "HIGH" (높은 참여도)
  elif score >= 50:      "MEDIUM" (중간 참여도)
  else:                  "LOW" (낮은 참여도)
```

#### 코드

```python
class EngagementAnalyzer:
    def analyze_frame(self, frame, tracked_objects):
        for x1, y1, x2, y2, student_id in tracked_objects:
            # ROI 추출
            roi = frame[y1:y2, x1:x2]
            
            # 움직임 분석
            motion = np.std(roi)
            
            # 점수 계산
            if motion > 30:
                engagement = 80
            elif motion > 15:
                engagement = 65
            elif motion > 5:
                engagement = 50
            else:
                engagement = 35
            
            # 밝기 보너스
            brightness = np.mean(roi)
            if brightness > 150:
                engagement += 10
            
            engagement = min(int(engagement), 100)
            
            # 데이터 저장
            self.student_data[student_id]['engagement_scores'].append(engagement)
            self.student_data[student_id]['motion'].append(motion)
```

#### 결과
- **입력**: 각 학생의 프레임별 이미지
- **출력**: 학생당 0-100 점수
- **예**: Student 1: 83/100 (HIGH), Student 2: 45/100 (LOW)

---

### STEP 4️⃣: 리포트 생성 (Report Generation)

#### JSON 형식

```json
{
  "metadata": {
    "video": "video_example.mp4",
    "frames_analyzed": 800,
    "fps": 30.0,
    "timestamp": "20251029_031023",
    "total_students": 32
  },
  "students": [
    {
      "student_id": 5,
      "frames_seen": 180,
      "avg_engagement": 83,
      "max_engagement": 83,
      "avg_motion": 33.4,
      "engagement_level": "high"
    },
    ...
  ]
}
```

#### HTML 형식
- 시각적 테이블 (색상 코딩)
- 참여도별 요약
- 학생별 상세 정보

---

## 🔍 예제: 실행 따라하기

### 1단계: 비디오 준비
```bash
# 프로젝트 디렉토리에 video_example.mp4 배치
ls -lh video_example.mp4
```

### 2단계: 스크립트 실행
```bash
python student_engagement_analysis.py
```

### 3단계: 결과 확인
```bash
# JSON 결과
cat outputs/reports/student_engagement_20251029_031023.json

# HTML 리포트 (브라우저에서 열기)
open outputs/reports/student_engagement_20251029_031023.html
```

---

## 📊 실제 결과

```
Found 32 unique students:
  Student  5: Engagement= 83/100 (high  ), Seen=180f, Motion= 33.4
  Student 22: Engagement= 82/100 (high  ), Seen=104f, Motion= 32.7
  Student  2: Engagement= 80/100 (high  ), Seen=170f, Motion= 32.3
  ...
  Student 27: Engagement= 53/100 (medium), Seen= 41f, Motion= 12.7
```

**분석**:
- 32명의 고유 학생 추적됨
- 평균 참여도: 71/100
- High: 20명, Medium: 12명
- 가장 참여도 높은 학생: 83/100
- 가장 참여도 낮은 학생: 53/100

---

## 🎓 학습 포인트

### 컴퓨터 비전 개념

1. **배경 차감 (Background Subtraction)**
   - MOG2 알고리즘
   - 픽셀 분류

2. **모폴로지 연산 (Morphological Operations)**
   - CLOSE: 구멍 채우기
   - OPEN: 노이즈 제거

3. **객체 추적 (Object Tracking)**
   - 중심점 거리 기반 매칭
   - 오클루션 처리

4. **특성 추출 (Feature Extraction)**
   - 움직임 분석
   - 밝기 분석

---

## ⚙️ 매개변수 조정

### 감지 임계값

```python
# ImprovedDetector.detect()

# 크기 필터
area_min = 5000      # 줄이면 작은 객체도 감지
area_max = 50000     # 늘리면 큰 객체도 감지

# 종횡비
aspect_ratio = 1.8   # 줄이면 더 많이 감지, 늘리면 엄격함

# 높이/너비
h_min = 50           # 너무 작은 것 제외
w_min = 20
```

### 추적 매개변수

```python
# StudentTracker.update()

best_dist = 250      # 최대 추적 거리 (픽셀)
max_age = 30         # 최대 미감지 프레임

# 줄이면: 추적이 더 엄격해짐
# 늘리면: 추적이 더 관대해짐
```

### 참여도 점수

```python
# EngagementAnalyzer.analyze_frame()

motion_high = 30     # High 임계값
motion_medium = 15   # Medium 임계값

brightness_threshold = 150

# 이 값들을 조정하면 점수 분포가 바뀜
```

---

## 🚀 실행 방법

```bash
# 기본 실행 (튜토리얼 주석 포함)
python student_engagement_analysis.py

# 결과 위치
outputs/reports/student_engagement_*.json
outputs/reports/student_engagement_*.html
```

---

## 📝 결론

이 시스템은 **4가지 핵심 기술**을 사용합니다:

1. ✅ **객체 감지**: MOG2 배경 차감
2. ✅ **다중 추적**: 중심점 매칭
3. ✅ **분석**: 움직임 + 밝기
4. ✅ **리포팅**: JSON + HTML

모든 단계가 **명확하게 주석**되어 있어 학습용으로 완벽합니다! 🎓
