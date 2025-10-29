# Classroom Engagement Analysis System - Implementation Guide

## 1. Installation & Setup

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA 11.0+) for YOLOv7 and DeepSORT
- Minimum 8GB RAM

### Step 1: Install Dependencies
```bash
cd ~/Desktop/videoanalysis
pip install -r requirements.txt

# For CUDA support (optional but recommended):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Download Pre-trained Models
```bash
# Create models directory
mkdir -p models

# YOLOv7 weights will auto-download on first use from torch.hub
# Alternatively, download manually:
# wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt -O models/yolov7.pt
```

### Step 3: Configure Gemini API
```bash
# Set environment variable with your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Or in .env file:
echo 'GEMINI_API_KEY="your-api-key-here"' > .env
```

### Step 4: Verify Installation
```python
python -c "
from src import PersonDetector, DeepSORTTracker, PoseEstimator
print('✓ All core modules imported successfully')
"
```

---

## 2. Configuration

Edit `config/config.yaml` to customize:

### Detection Settings
```yaml
detection:
  confidence_threshold: 0.45  # Lower = more detections (higher false positives)
  nms_threshold: 0.5          # Duplicate suppression
  device: "cuda"              # or "cpu"
```

### Tracking Settings
```yaml
tracking:
  max_age: 30        # Frames to keep track without detection
  min_hits: 3        # Min detections before confirming ID
  nn_budget: 100     # Memory for appearance features
```

### Feature Engineering
```yaml
features:
  window_size: 3.0                    # Aggregation window (seconds)
  gaze:
    eye_contact_threshold_deg: 30     # Threshold for "engaged" gaze
  posture:
    stability_variance_threshold_deg: 5  # Threshold for stability
```

### Engagement Classification Weights
```yaml
classification:
  rule_based:
    weights:
      gaze: 0.40           # Gaze importance (40%)
      posture: 0.20        # Posture importance (20%)
      gesture: 0.20        # Hand activity importance (20%)
      interaction: 0.20    # Device interaction importance (20%)
```

---

## 3. Basic Usage

### Minimal Example
```python
import yaml
import cv2
from src import PersonDetector, DeepSORTTracker, PoseEstimator, FeatureExtractor, EngagementClassifier

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize components
detector = PersonDetector(**config['detection'])
tracker = DeepSORTTracker(**config['tracking'])
pose_estimator = PoseEstimator(**config['pose_estimation']['mediapipe'])
feature_extractor = FeatureExtractor(config)
classifier = EngagementClassifier(config)

# Process video
video_path = 'data/raw/classroom.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
results_timeline = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect persons
    detections = detector.detect(frame)
    
    # Track IDs
    tracked_objects = tracker.update(detections, frame_idx, frame)
    
    # Estimate pose for each tracked person
    pose_results = {}
    for obj in tracked_objects:
        track_id = obj['track_id']
        bbox = obj['bbox']
        x_min, y_min, x_max, y_max = bbox
        cropped = frame[y_min:y_max, x_min:x_max]
        pose_results[track_id] = pose_estimator.estimate(cropped)
    
    # Extract engagement features
    timestamp = frame_idx / fps
    frame_features = feature_extractor.extract_per_frame(
        frame_idx=frame_idx,
        timestamp_sec=timestamp,
        tracked_objects=tracked_objects,
        pose_results=pose_results,
        fps=fps,
    )
    
    # Classify engagement
    frame_classifications = {}
    for student_id, features in frame_features.items():
        frame_classifications[student_id] = classifier.classify(features)
    
    results_timeline.append({
        'frame_idx': frame_idx,
        'timestamp': timestamp,
        'detections': len(tracked_objects),
        'classifications': frame_classifications,
    })
    
    frame_idx += 1
    
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
print(f"Processing complete. Total frames: {frame_idx}")
```

---

## 4. Advanced: End-to-End Pipeline

Create `src/pipeline.py`:

```python
import yaml
import numpy as np
import pandas as pd
from src import (
    PersonDetector, DeepSORTTracker, PoseEstimator,
    FeatureExtractor, EngagementClassifier, ReportGenerator
)

class EngagementAnalysisPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.detector = PersonDetector(**self.config['detection'])
        self.tracker = DeepSORTTracker(**self.config['tracking'])
        self.pose_estimator = PoseEstimator(**self.config['pose_estimation']['mediapipe'])
        self.feature_extractor = FeatureExtractor(self.config)
        self.classifier = EngagementClassifier(self.config)
        self.report_generator = ReportGenerator(self.config)
    
    def process_video(self, video_path: str, output_dir: str = "outputs/"):
        """Process video and return structured results."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        timeline = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection → Tracking → Pose → Features → Classification
            detections = self.detector.detect(frame)
            tracked_objects = self.tracker.update(detections, frame_idx, frame)
            
            pose_results = {}
            for obj in tracked_objects:
                x_min, y_min, x_max, y_max = obj['bbox']
                cropped = frame[y_min:y_max, x_min:x_max]
                pose_results[obj['track_id']] = self.pose_estimator.estimate(cropped)
            
            frame_features = self.feature_extractor.extract_per_frame(
                frame_idx, frame_idx / fps, tracked_objects, pose_results, fps
            )
            
            for student_id, features in frame_features.items():
                classification = self.classifier.classify(features)
                timeline.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'student_id': student_id,
                    'features': features,
                    'classification': classification,
                })
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
        
        cap.release()
        
        return {
            'video_path': video_path,
            'total_frames': frame_idx,
            'fps': fps,
            'timeline': timeline,
        }
    
    def generate_report(self, results: dict, output_format: str = "text"):
        """Generate final report."""
        session_data = {
            'session_name': results['video_path'],
            'session_duration_min': results['total_frames'] / results['fps'] / 60,
            'students_tracked': list(set(r['student_id'] for r in results['timeline'])),
            'timeline': results['timeline'],
        }
        
        return self.report_generator.generate_report(session_data, output_format)

# Usage:
# pipeline = EngagementAnalysisPipeline('config/config.yaml')
# results = pipeline.process_video('data/raw/classroom_video.mp4')
# report = pipeline.generate_report(results, output_format='html')
```

---

## 5. Output Interpretation

### Engagement Score Breakdown

Each student receives:
- **Engagement Score** (0-100):
  - 0-33: Disengaged (low attention, minimal activity)
  - 34-66: Passive (moderate attention, limited interaction)
  - 67-100: Engaged (high attention, active participation)

- **Component Scores**:
  - **Gaze Score**: Based on head direction (0°=facing screen → 100 points)
  - **Posture Score**: Based on stability (low variance → higher score)
  - **Gesture Score**: Based on hand activity frequency
  - **Interaction Score**: Based on device/tablet proximity

### Example Report Output

```
================================================================================
CLASSROOM ENGAGEMENT ANALYSIS REPORT
================================================================================

Session: classroom_session_001.mp4
Date: 2025-10-29 14:32:00
Duration: 45 minutes
Students Tracked: 28

--------------------------------------------------------------------------------
ENGAGEMENT TIMELINE
--------------------------------------------------------------------------------

[10:00-10:03] S001 (Score: 78/100, engaged)
  Gaze: 15.2° from screen → HIGH (facing screen) (score: 86.1)
  Posture Stability: σ=3.2° → STABLE (upright seated) (score: 92.5)
  Hand Gestures: 2.1 events/min → ACTIVE (score: 42.0)
  Device Interaction: 0.67 → ACTIVE (hands on device) (score: 67.0)
  
  [Narrative from Gemini]
  Student maintained forward-facing gaze with stable posture and regular hand 
  movements indicating active engagement with tablet-based task.

[10:03-10:06] S001 (Score: 52/100, passive)
  Gaze: 42.8° from screen → MODERATE (partial attention) (score: 52.3)
  ...

================================================================================
STATISTICAL SUMMARY
================================================================================

Mean Engagement: 62.3/100
Std Deviation: 18.7
Min Score: 12.5
Max Score: 95.8
```

---

## 6. Validation & Accuracy

### Recommended Validation Steps

1. **Manual Annotation** (for model calibration):
   - Label 100-200 sample frames with engagement level (3 human raters)
   - Compute inter-rater agreement (Fleiss' κ)
   - Validate system predictions against consensus labels

2. **Pose Estimation Accuracy**:
   - Compare MediaPipe keypoints against manual/OptiTrack ground truth
   - Compute PCK (Percentage of Correct Keypoints) per joint
   - Target: PCK@0.1 > 80% for key joints (nose, shoulders, wrists)

3. **Tracking Stability**:
   - Monitor ID switches (false reassignments)
   - Track length distributions
   - Target: <5% ID switches per 100 frames

4. **Engagement Classification**:
   - Confusion matrix: Precision/Recall/F1-score per level
   - Spearman correlation with manual ratings
   - Target: ρ > 0.75 with manual labels

---

## 7. Performance Optimization

### For Real-Time Processing
```yaml
detection:
  confidence_threshold: 0.50  # Slightly higher for speed
  device: "cuda"

pose_estimation:
  mediapipe:
    model_complexity: 0  # Lite model (faster, less accurate)

features:
  window_size: 1.0  # Shorter window for responsiveness
```

### For High-Accuracy Offline Analysis
```yaml
detection:
  confidence_threshold: 0.40  # Lower for comprehensive detection
  device: "cuda"

pose_estimation:
  mediapipe:
    model_complexity: 2  # Full model (slower, more accurate)

features:
  window_size: 5.0  # Longer window for stability
```

### Benchmarks (on RTX 3090)
- YOLOv7: ~30ms/frame
- MediaPipe Pose: ~15ms/frame
- DeepSORT: ~5ms/frame
- Feature extraction: ~2ms/frame
- **Total: ~52ms/frame → ~19 FPS**

---

## 8. Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size or use CPU
- Use model_complexity: 0 for pose estimation

**"Model weights not found"**
- Ensure `models/` directory exists
- YOLOv7 will auto-download on first use (requires internet)

**"Gemini API errors"**
- Verify GEMINI_API_KEY is set correctly
- Check internet connection
- Report generation will fall back to rule-based narratives

**"Poor tracking (IDs switch frequently)"**
- Increase `tracking.max_age` (allows longer gaps)
- Increase `tracking.nn_budget` (better feature memory)
- Ensure sufficient lighting in classroom

**"Low engagement scores for all students"**
- Check gaze threshold: `features.gaze.eye_contact_threshold_deg`
- Verify camera angle (should be frontal)
- Adjust feature weights in `classification.rule_based.weights`

---

## 9. Citation & References

For academic papers, cite:
```bibtex
@software{classroom_engagement_analyzer_2025,
  title={Classroom Engagement Analysis System: 
         AI-Powered Student Activity Monitoring},
  author={[Your Name/Lab]},
  year={2025},
  url={https://github.com/your-org/videoanalysis}
}
```

Key dependencies:
- YOLOv7: Wang et al. (2022) - https://arxiv.org/abs/2207.02696
- DeepSORT: Wojke et al. (2017) - https://arxiv.org/abs/1703.07402
- MediaPipe: Lugaresi et al. (2019) - https://arxiv.org/abs/1906.08172

---

## 10. Next Steps

1. ✓ Install dependencies
2. ✓ Download models
3. ✓ Configure system (edit config.yaml)
4. → Prepare sample classroom video
5. → Run end-to-end pipeline
6. → Validate against manual annotations
7. → Generate reports for teachers
8. → Iterate on feature thresholds

For questions or issues, refer to README.md or the inline documentation in source code.
