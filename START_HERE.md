# 🎓 Classroom Engagement Analysis System
## START HERE - 빠른 시작 가이드

---

## 📋 시스템 구성

이 프로젝트는 **학급 참여도 분석**을 위한 학술 수준의 AI 시스템입니다.

```
입력 (Video/Audio)
    ↓
감지 (YOLOv7) → 학생 인식
    ↓
추적 (DeepSORT) → 학생 ID 유지
    ↓
자세 추정 (MediaPipe) → 머리/손/몸 위치
    ↓
특징 추출 → 시선/자세/손/상호작용 신호
    ↓
분류 (Rule-based) → 0-100 참여도 점수
    ↓
리포트 생성 (Gemini API) → 설명 가능한 결과
```

---

## ⚡ 5분 안에 시작하기

### Step 1: 설치
```bash
cd ~/Desktop/videoanalysis
pip install -r requirements.txt
```

### Step 2: 비디오 준비
```bash
mkdir -p data/raw outputs/reports outputs/visualizations
cp /path/to/video_example.mp4 data/raw/
```

### Step 3: 빠른 테스트 (30프레임)
```bash
python -c "
import yaml
import cv2
from src import PersonDetector, EngagementClassifier

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

detector = PersonDetector('models/yolov7.pt', device='cpu', half_precision=False)
classifier = EngagementClassifier(config)

cap = cv2.VideoCapture('data/raw/video_example.mp4')
for i in range(30):
    ret, frame = cap.read()
    if ret:
        detections = detector.detect(frame)
        print(f'Frame {i}: {len(detections)} persons detected')

print('✓ 시스템이 정상 작동합니다!')
"
```

### Step 4: 전체 분석
```bash
# 제공된 스크립트 실행
python analyze_video_full.py

# 결과는 outputs/reports/ 에 저장됨
ls -lh outputs/reports/
```

---

## 📊 주요 출력물

분석 후 다음 파일들이 생성됩니다:

| 파일 | 형식 | 용도 |
|------|------|------|
| `engagement_report_*.txt` | 텍스트 | 콘솔/메모 |
| `engagement_report_*.html` | 웹 페이지 | 교사 보고서 (시각적) |
| `engagement_data_*.json` | 데이터 | 추가 분석용 구조화 데이터 |
| `engagement_analysis.png` | 그래프 | 시각화 요약 |

---

## 🎯 핵심 개념

### 참여도 점수 (Engagement Score)
```
0-33:   Low (❌ 낮은 참여도 - 주의 필요)
34-66:  Moderate (⚠️ 보통 - 개선 권고)
67-100: High (✓ 높은 참여도 - 좋음)
```

### 4가지 측정 신호

| 신호 | 설명 | 가중치 |
|------|------|--------|
| 시선 (Gaze) | 화면 방향 각도 | 40% |
| 자세 (Posture) | 어깨-엉덩이 기울기 안정성 | 20% |
| 손 (Gesture) | 손짓/필기 빈도 | 20% |
| 상호작용 (Interaction) | 태블릿 근처 거리 | 20% |

모든 점수는 **완전히 해석 가능**하며 특정 수치에 기반합니다.

---

## 🔧 주요 파일 및 역할

```
videoanalysis/
├── config/config.yaml              ← 모든 설정 (임계값, 가중치, 모델)
├── src/
│   ├── detector.py                 ← YOLOv7 (사람 감지)
│   ├── tracker.py                  ← DeepSORT (ID 유지)
│   ├── pose_estimator.py           ← MediaPipe (자세 추정)
│   ├── feature_engineering.py      ← 신호 계산
│   ├── engagement_classifier.py    ← 점수화
│   └── report_generator.py         ← Gemini 리포트
├── README.md                        ← 기술 상세 문서
├── IMPLEMENTATION_GUIDE.md          ← 구현 가이드
├── TEST_GUIDE_VIDEO.md             ← 비디오 테스트 가이드
└── START_HERE.md                   ← 이 문서
```

---

## 📈 예상 성능

| 구성 | 처리 속도 | GPU | 정확도 |
|------|---------|-----|--------|
| **CPU (기본)** | ~5 FPS | 없음 | 좋음 |
| **GPU (권장)** | ~19 FPS | CUDA | 우수 |
| **경량 설정** | ~30 FPS | CUDA | 보통 |

---

## ❓ 자주 묻는 질문

### Q1: 왜 사람을 감지하지 못할까?
**A**: 다음을 확인하세요:
- 비디오 해상도 (최소 480p)
- 조명 상태 (충분한 밝기)
- config.yaml의 `confidence_threshold` 낮추기 (0.40으로)

### Q2: 참여도 점수가 항상 낮음
**A**: 설정 조정:
```yaml
# config/config.yaml
features:
  gaze:
    eye_contact_threshold_deg: 40  # 기본 30도에서 올리기
  posture:
    stability_variance_threshold_deg: 8  # 기본 5도에서 올리기
```

### Q3: Gemini API가 필요한가?
**A**: 아니오. 없어도 규칙 기반 설명 자동 생성. API 있으면 더 자연스러운 문장.

### Q4: 실시간 처리 가능한가?
**A**: GPU에서 ~19 FPS. 교실용 30fps 영상은 약간 지연 (0.1초).

### Q5: 개인정보는 안전한가?
**A**: 
- ✓ 얼굴 인식 안 함
- ✓ 신원 추정 안 함  
- ✓ 감정 판단 안 함
- ✓ 데이터는 행동 신호만 저장

---

## 🚀 다음 단계

### 입문자
1. ✓ START_HERE.md 읽기 (지금)
2. → TEST_GUIDE_VIDEO.md로 첫 비디오 테스트
3. → 결과 검토 및 config 조정

### 중급 사용자
1. ✓ IMPLEMENTATION_GUIDE.md 숙지
2. → 여러 비디오로 시스템 검증
3. → 클래스별 매개변수 최적화
4. → 교사 피드백 반영

### 개발자/연구자
1. ✓ README.md 기술 세부사항 확인
2. → src/ 모듈 커스터마이징
3. → 새로운 특징 추가
4. → ML 모델 훈련 (선택사항)

---

## 📞 트러블슈팅 체크리스트

```
[ ] Python 3.8+ 설치?
    → python --version

[ ] 필수 패키지 설치?
    → pip install -r requirements.txt

[ ] 비디오 파일 존재?
    → ls -lh data/raw/video_example.mp4

[ ] config.yaml 확인?
    → cat config/config.yaml | head -20

[ ] 경고/에러 없음?
    → python test_video_quick.py 2>&1 | grep -i error

[ ] 사람 감지됨?
    → 해상도 >= 480p, 조명 충분한지 확인

[ ] 점수가 합리적인가?
    → 0-100 범위, 0-33 낮음/34-66 보통/67-100 높음
```

---

## 📖 상세 문서 맵

```
START_HERE.md (지금 읽는 곳)
    ↓
TEST_GUIDE_VIDEO.md ← 실제 비디오 테스트하고 싶을 때
    ↓
IMPLEMENTATION_GUIDE.md ← 완전한 구현 및 배포
    ↓
README.md ← 기술 깊이 있는 설명
    ↓
src/*/code ← 소스 코드 및 인라인 문서
```

---

## 💡 팁

### 빠른 성능 테스트
```bash
# 첫 100프레임만 처리 (디버깅용)
python -c "
import cv2
from src import PersonDetector

detector = PersonDetector('models/yolov7.pt', device='cpu', half_precision=False)
cap = cv2.VideoCapture('data/raw/video_example.mp4')

for i in range(100):
    ret, frame = cap.read()
    if ret:
        detections = detector.detect(frame)
        print(f'Frame {i}: {len(detections)} detected')
"
```

### 특정 시간대 분석
```bash
# 10-20초만 처리
python -c "
import cv2
fps = 30
cap = cv2.VideoCapture('data/raw/video_example.mp4')
start_frame = int(10 * fps)
end_frame = int(20 * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for i in range(start_frame, end_frame):
    ret, frame = cap.read()
    # 분석...
"
```

---

## ✅ 작동 확인 체크리스트

시스템이 올바르게 설치되었는지 확인:

```bash
# 1. 패키지 임포트
python -c "from src import PersonDetector, EngagementClassifier; print('✓ 패키지 OK')"

# 2. config 로드
python -c "import yaml; yaml.safe_load(open('config/config.yaml')); print('✓ 설정 OK')"

# 3. 비디오 열기
python -c "import cv2; cv2.VideoCapture('data/raw/video_example.mp4').isOpened() and print('✓ 비디오 OK')"

# 4. 모든 것 종합
python test_video_quick.py
```

모두 ✓이면 준비 완료!

---

## 📝 License & Citation

학술 목적 인용:
```bibtex
@software{classroom_engagement_2025,
  title={Classroom Engagement Analysis System},
  author={Your Name},
  year={2025},
  url={https://github.com/...}
}
```

---

**이제 TEST_GUIDE_VIDEO.md로 진행하세요! 🚀**
