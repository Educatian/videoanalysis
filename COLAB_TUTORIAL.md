# Google Colab으로 Classroom Engagement Analysis 실행하기

## 🚀 5분 안에 시작하기

### Step 1: Google Colab 열기

1. 아래 링크를 클릭하면 자동으로 Google Colab에서 열립니다:
   - https://colab.research.google.com/

2. 새로운 노트북을 생성합니다

### Step 2: 코드 복사 및 실행

다음 코드를 **Colab의 첫 번째 셀**에 붙여넣습니다:

```python
# Clone repository and setup
!git clone https://github.com/Educatian/videoanalysis.git
%cd videoanalysis

# Install dependencies
!pip install opencv-python numpy -q

# Run analysis
!python classroom_engagement_colab.py
```

**Shift + Enter** 를 눌러 셀을 실행합니다.

### Step 3: 비디오 파일 업로드

스크립트가 파일 업로드를 요청할 때:
1. **"Choose Files"** 버튼 클릭
2. `video_example.mp4` 또는 분석할 비디오 파일 선택
3. 분석 자동 시작

### Step 4: 결과 다운로드

분석 완료 후:
```python
# 다음 셀에서 실행
from google.colab import files

# 결과 다운로드
files.download('outputs/reports/video_analysis_*.html')
files.download('outputs/reports/video_analysis_*.json')
```

---

## 📋 전체 Colab 노트북 코드

### 셀 1: 저장소 복제 및 환경 설정

```python
# Repository 복제
!git clone https://github.com/Educatian/videoanalysis.git
%cd videoanalysis

# 의존성 설치 (Colab은 OpenCV, NumPy 사전 설치)
!pip install opencv-python numpy -q

print("Environment setup complete!")
```

### 셀 2: 분석 스크립트 실행

```python
# 메인 분석 스크립트 실행
!python classroom_engagement_colab.py
```

### 셀 3: 결과 다운로드

```python
from google.colab import files
from pathlib import Path

# 최신 분석 결과 찾기
reports_dir = Path('outputs/reports')
files_list = sorted(reports_dir.glob('video_analysis_*.html'))

if files_list:
    latest_html = files_list[-1]
    print(f"Downloading: {latest_html}")
    files.download(str(latest_html))
    
    latest_json = str(latest_html).replace('.html', '.json')
    print(f"Downloading: {latest_json}")
    files.download(latest_json)
else:
    print("No analysis files found")
```

### 셀 4: 결과 시각화 (선택사항)

```python
import json
import pandas as pd
from pathlib import Path

# 최신 JSON 결과 로드
results_file = sorted(Path('outputs/reports').glob('*.json'))[-1]

with open(results_file) as f:
    data = json.load(f)

# 통계 출력
stats = data['statistics']
print("=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)
print(f"Frames Analyzed: {stats['num_frames']}")
print(f"Avg Engagement: {stats['engagement']['mean']:.1f}/100")
print(f"Engagement Std Dev: {stats['engagement']['std']:.1f}")
print(f"Avg Objects/Frame: {stats['objects']['avg_per_frame']:.1f}")
print("=" * 60)
```

### 셀 5: 프레임별 상세 분석 (선택사항)

```python
# 첫 20프레임 상세 데이터 출력
frame_data = data['frame_data']

print("FRAME-BY-FRAME ANALYSIS (First 20 frames)")
print("-" * 80)
print(f"{'Frame':<6} {'Time':<8} {'Motion':<8} {'Objects':<8} {'Score':<7} {'Level':<10}")
print("-" * 80)

for frame in frame_data:
    print(f"{frame['frame']:<6} "
          f"{frame['timestamp']:<8.2f} "
          f"{frame['motion_intensity']:<8.1f} "
          f"{frame['num_objects']:<8} "
          f"{frame['engagement_score']:<7} "
          f"{frame['engagement_level']:<10}")
```

---

## 🎯 주요 기능

### 분석 알고리즘

1. **배경 차감 (MOG2)**
   - 움직이는 객체 감지
   - 조명 변화에 강건

2. **모션 강도 계산**
   - 움직이는 픽셀 비율 (%)
   - 0-100% 범위

3. **객체 감지**
   - 컨투어 분석
   - 크기별 필터링 (500-100,000px)

4. **참여도 점수**
   - 모션 + 객체 수 기반
   - 0-100점 척도
   - 3단계 분류: High/Moderate/Low

### 출력 형식

#### HTML 리포트
- 아름다운 웹 기반 UI
- 통계 요약
- 프레임별 데이터 테이블
- 즉시 열람 가능

#### JSON 데이터
- 기계 가독형 형식
- 구조화된 메타데이터
- 프레임별 상세 데이터
- 추가 분석용

---

## ⚠️ 주의사항

### Colab 리소스 제한
- **메모리**: 12GB (대부분의 비디오에 충분)
- **저장공간**: 100GB (Google Drive 마운트 권장)
- **시간 제한**: 12시간 (장시간 분석은 Google Drive 마운트 권장)

### 장시간 분석 (>1시간)

Google Drive를 마운트하여 결과를 영구 저장:

```python
from google.colab import drive
import os

# Google Drive 마운트
drive.mount('/content/drive')

# 작업 디렉토리 변경
os.chdir('/content/drive/My Drive')
```

---

## 🔧 문제 해결

### 문제 1: "ModuleNotFoundError"

**해결책**: 모든 의존성이 설치되었는지 확인
```python
!pip install opencv-python numpy scikit-image -q
```

### 문제 2: 비디오 업로드 불가

**해결책**: 대용량 파일은 Google Drive 마운트 사용
```python
from google.colab import drive
drive.mount('/content/drive')

# Drive에서 파일 사용
video_path = '/content/drive/My Drive/video_example.mp4'
```

### 문제 3: 메모리 부족

**해결책**: 분석 프레임 수 감소
```python
# 첫 50프레임만 분석
analyze_video(video_path, max_frames=50)
```

---

## 📊 예상 실행 시간

| 프레임 수 | 예상 시간 | 파일 크기 |
|---------|---------|---------|
| 50프레임 | ~30초 | ~100KB |
| 100프레임 | ~1분 | ~200KB |
| 500프레임 | ~5분 | ~1MB |
| 1000프레임 | ~10분 | ~2MB |

---

## 💾 Google Drive에 결과 저장

```python
from google.colab import drive
from pathlib import Path
import shutil

# Google Drive 마운트
drive.mount('/content/drive')

# 결과를 Drive로 복사
src_dir = Path('outputs/reports')
dst_dir = Path('/content/drive/My Drive/engagement_results')

dst_dir.mkdir(exist_ok=True)

for file in src_dir.glob('*'):
    shutil.copy(str(file), str(dst_dir / file.name))
    print(f"Saved: {file.name}")
```

---

## 📚 참고 자료

- **GitHub 저장소**: https://github.com/Educatian/videoanalysis
- **OpenCV 문서**: https://docs.opencv.org/
- **Google Colab**: https://colab.research.google.com/

---

## ✅ Colab 체크리스트

- [ ] Colab 노트북 생성
- [ ] 첫 번째 셀 실행 (저장소 복제)
- [ ] 두 번째 셀 실행 (분석 스크립트)
- [ ] 비디오 파일 업로드
- [ ] 분석 완료 대기
- [ ] 결과 다운로드
- [ ] HTML 리포트 열기
- [ ] JSON 데이터 검토

---

## 🎉 완료!

분석이 완료되면:
1. **HTML 리포트**: 웹 브라우저에서 열어 시각적 확인
2. **JSON 데이터**: Excel/Python으로 추가 분석
3. **결과 공유**: GitHub 또는 Google Drive에 저장

**모든 분석이 클라우드에서 자동으로 처리됩니다!** ☁️
