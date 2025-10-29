# ⚡ 빠른 시작 가이드

## 즉시 실행하기 (지금 바로!)

```bash
# 1단계: 프로젝트 폴더 열기
cd videoanalysis

# 2단계: 간단한 분석 실행
python run_simple_analysis.py

# 또는 더 자세한 분석
python analyze_video_opencv.py
```

**결과**: `outputs/reports/` 폴더에 JSON과 HTML 리포트 생성

---

## 📊 3가지 분석 방법

### 1️⃣ 초고속 테스트 (권장) ⭐
```bash
python run_simple_analysis.py
```
- **소요시간**: 1초
- **기능**: 모션 감지, 참여도 점수
- **필요 패키지**: OpenCV, NumPy
- **상태**: ✅ **즉시 작동**

### 2️⃣ 표준 분석
```bash
python analyze_video_opencv.py
```
- **소요시간**: 5초 (100프레임)
- **기능**: 객체 감지, 모션 분석, 통계
- **필요 패키지**: OpenCV, NumPy
- **상태**: ✅ **즉시 작동**

### 3️⃣ 클라우드 분석 (Google Colab)
```bash
# Colab 노트북에서:
!git clone https://github.com/Educatian/videoanalysis.git
%cd videoanalysis
!python classroom_engagement_colab.py
```
- **장점**: 모든 의존성 미리 설치됨
- **상태**: ✅ **작동 보장**

---

## 🔧 문제 해결

| 문제 | 해결책 |
|------|--------|
| `ModuleNotFoundError` | `python run_simple_analysis.py` 사용 |
| `YOLOv7 로드 실패` | OpenCV 버전 사용 또는 Colab 사용 |
| `MediaPipe 호환성` | Python 3.10 이하 사용 권장 |
| `DeepSORT 없음` | 자동 스킵됨, 기본 기능은 작동 |

자세한 내용은 `DEBUG_REPORT.md` 참조

---

## 📁 프로젝트 구조

```
videoanalysis/
├── run_simple_analysis.py          ← 초고속 분석
├── analyze_video_opencv.py         ← 표준 분석
├── classroom_engagement_colab.py   ← Colab 분석
├── test_video_quick_run.py         ← 고급 기능 (의존성 필요)
├── video_example.mp4               ← 테스트 비디오
├── config/
│   └── config.yaml                 ← 설정 파일
├── src/
│   ├── detector.py                 ← YOLOv7
│   ├── tracker.py                  ← DeepSORT
│   ├── pose_estimator.py           ← MediaPipe
│   ├── feature_engineering.py      ← 기능 추출
│   ├── engagement_classifier.py    ← 분류기
│   └── report_generator.py         ← 리포트 생성
└── outputs/
    └── reports/                    ← 분석 결과
```

---

## 🎯 다음 단계

### 기본 사용
1. ✅ `python run_simple_analysis.py` 실행
2. 📊 `outputs/reports/` 에서 결과 확인

### 고급 기능 (선택사항)
```bash
# 모든 기능 활성화
pip install -r requirements.txt
python test_video_quick_run.py
```

### 커스터마이징
- `config/config.yaml` 수정
- `src/` 폴더의 스크립트 수정
- GitHub에 커밋 후 푸시

---

## 📞 문의 & 참고

- **GitHub**: https://github.com/Educatian/videoanalysis
- **문제 해결**: `DEBUG_REPORT.md` 참조
- **상세 가이드**: `IMPLEMENTATION_GUIDE.md` 참조
- **학술 정보**: `README.md` 참조

---

## ✅ 현재 상태

| 기능 | 상태 | 스크립트 |
|------|------|---------|
| 비디오 로드 | ✅ | 모두 |
| 모션 감지 | ✅ | 모두 |
| 객체 감지 | ✅ | run_simple, analyze |
| 포즈 추정 | ⚠️ | 고급 only |
| 객체 추적 | ⚠️ | 고급 only |
| 참여도 점수 | ✅ | 모두 |
| 리포트 생성 | ✅ | 모두 |

**권장**: OpenCV 버전부터 시작하기! 🚀
