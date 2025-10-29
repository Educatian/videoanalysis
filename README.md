# Classroom Engagement Analysis System
## AI-Powered Student Engagement Tracking with Explainable Features

### Overview
This system provides **academically rigorous, scientifically grounded analysis** of student engagement in classroom settings using computer vision, pose estimation, and explainable AI. It combines multi-modal input streams (video, audio) with advanced tracking and feature engineering to generate teacher-actionable insights.

**Key Research Design Principles:**
- **Transparency**: All engagement scores backed by quantifiable, interpretable features
- **Objectivity**: Rule-based and ML-based classifications with documented thresholds
- **Multi-modal**: Integration of visual cues (pose, gaze, gestures) rather than single signals
- **Ethical**: Privacy-preserving, avoids sensitive identity/personal attribute inference

---

## System Architecture

### 1. Input & Synchronization Module
**Purpose**: Aggregate heterogeneous classroom data streams with temporal alignment

#### Data Sources
- **Camera 1 (Full-body overview)**: Broad classroom scene (person detection baseline)
- **Camera 2 (Overhead desk/tablet)**: Task-specific interaction capture
- **Audio (Lapel microphone)**: Speech/engagement activity correlation (optional)

#### Synchronization Strategy
- Extract frame timestamps from video metadata (fps, frame_idx)
- Align audio via frame-to-time mapping: `t_frame = frame_idx / fps`
- Store synchronized data as HDF5 with inter-modal cross-references

---

### 2. Detection & Tracking (DeepSORT + YOLOv7)

#### Person Detection
**Model**: YOLOv7 (COCO pretrained)
- Input: Raw video frames, 640×640 or adaptive resolution
- Output: Bounding boxes [x_min, y_min, x_max, y_max] with confidence scores
- Threshold: confidence ≥ 0.45 (tunable for precision/recall tradeoff)

#### Multi-Object Tracking
**Method**: DeepSORT (Deep Convolutional Sort)
- Maintains stable **Student ID** (`ID_student_1`, `ID_student_2`, ...) across frames
- Uses appearance features (CNN embeddings from detection backbone)
- Hungarian algorithm for frame-to-frame association
- Kalman filtering for motion prediction (handles occlusion/camera shake)

**Output**: Frame-level tracking data
```
frame_id | timestamp | student_id | bbox | confidence | age (frames)
```

---

### 3. Pose & Landmark Estimation

#### Option A: OpenPose (High Precision)
- 25-joint skeleton model (COCO+Foot keypoints)
- Heavier computational cost (~50-100ms/frame on GPU)
- **Recommended for**: Offline analysis, high-accuracy requirements

#### Option B: MediaPipe (Real-Time Optimized)
- **Pose**: 33 whole-body landmarks (with depth inference on single image)
- **Hand**: 21 landmarks per hand (bilateral)
- **Face Landmarker** (optional): 468 face keypoints + head pose angles
- Lightweight & fast (~10-20ms/frame on CPU)
- **Recommended for**: Real-time systems, classroom deployment

**Keypoint Convention** (MediaPipe Pose):
```
Torso:        0(nose), 1(left_eye), 2(right_eye), 3(left_ear), 4(right_ear),
              5(left_shoulder), 6(right_shoulder), 7(left_elbow), 8(right_elbow),
              9(left_wrist), 10(right_wrist), 11(left_hip), 12(right_hip)
Lower Body:   13-16 (knees/ankles), 17-21 (feet details)
```

**Confidence & Visibility**: Each landmark has `(x, y, z, confidence)` triplet
- `confidence > 0.5` used for filtering unreliable detections

---

### 4. Feature Engineering for Engagement Classification

#### 4.1 Head Pose & Gaze Direction
**Mathematical Definition**:
- Extract head keypoints: `P_nose`, `P_left_ear`, `P_right_ear`
- Compute head normal vector: `v_head = (P_left_ear - P_right_ear) × (P_nose - midpoint(P_left_ear, P_right_ear))`
- Screen direction: `v_screen = normalize([screen_center_x - P_nose.x, -P_nose.y])`  
- **Engagement angle**: `θ_gaze = arccos(v_head · v_screen / (||v_head|| · ||v_screen||))`
  - `θ < 30°`: High engagement (facing screen/teacher)
  - `30° ≤ θ < 60°`: Moderate engagement
  - `θ ≥ 60°`: Low engagement (looking away)

#### 4.2 Posture Stability
**Upper Body Tilt**:
- Compute shoulder-hip axis: `v_posture = P_right_hip - P_right_shoulder`
- Tilt angle from vertical: `φ = arctan(v_posture.x / v_posture.y)`
- **Stability metric**: Variance of `φ` over sliding window (2-5 sec)
  - Low variance (σ < 5°): Stable seated posture → +engagement
  - High variance (σ > 15°): Fidgeting/instability → -engagement

#### 4.3 Hand Activity & Gesture Detection
**Wrist Motion Velocity**:
- Temporal differentiation: `v_wrist = (P_wrist(t) - P_wrist(t-1)) / Δt`
- Cumulative motion: `motion_accum = Σ||v_wrist||` over time window
- **Interpretation**: 
  - Writing (tablet): rapid, localized motion clusters
  - Pointing: ballistic, directional motion
  - Idle: low motion

**Gesture Classification**:
- **Pointing gesture**: `v_hand · v_screen > 0.7` (hand direction aligned with screen)
- **Writing gesture**: Motion concentrated in small region (variance of hand position)
- **Frequency**: Event count per time window (gestures/min)

#### 4.4 Task-Specific Interaction
**Screen/Tablet Interaction**:
- Compute hand-to-tablet vector: `v_interact = P_tablet_center - P_wrist`
- Proximity: Distance from wrist to tablet surface
- **Interaction index**: Product of proximity score × gesture frequency

#### 4.5 Time-Windowed Aggregation
All features aggregated over sliding window (default: 3 seconds, 90% overlap):
- Per-student summary: `{θ_gaze, σ_posture, f_gesture, I_interact, ...}`
- Standardization (z-score normalization) for cross-student comparability

---

### 5. Engagement Classification

#### Rule-Based Scoring (Default)
```
engagement_score = (
    w_gaze * f_gaze(θ) +
    w_posture * f_posture(σ) +
    w_gesture * f_gesture(freq) +
    w_interact * f_interact(I)
)
```

Where:
- `f_gaze(θ) = max(0, 1 - θ/90)`: Linear decay with gaze angle
- `f_posture(σ) = exp(-σ²/50)`: Gaussian penalty for instability
- `f_gesture(freq) = min(1, freq/5)`: Saturating function (normalized by expected gesture rate)
- `f_interact(I) = min(1, I/max_I)`: Normalized interaction index
- Weights: `w_gaze=0.4, w_posture=0.2, w_gesture=0.2, w_interact=0.2` (tunable per classroom)

**Output**: Engagement Score ∈ [0, 100]
- 0-33: Low engagement (disengaged)
- 34-66: Moderate engagement (on-task)
- 67-100: High engagement (focused/interactive)

#### ML-Based Classification (Optional)
For improved accuracy, train lightweight classifier (LogisticRegression or XGBoost):
- **Input features**: `[θ_gaze, σ_posture, f_gesture, I_interact, temporal_features]`
- **Target variable**: Manual annotation (3-5 label categories: disengaged, passive, engaged, highly-engaged, confused)
- **Train/Val split**: 70/30 on pilot classroom data
- **Cross-validation**: 5-fold to assess robustness

---

### 6. Report Generation with Gemini API

#### Methodology
1. **Feature aggregation**: Summarize per-student metrics over report period
2. **Anomaly detection**: Identify time intervals with engagement deviations
3. **LLM-based narrative**: Pass structured feature data to Gemini for explanation
4. **Quality control**: Enforce ethical constraints via system prompt

#### Gemini System Prompt (Safety)
```
[System Prompt]
You are an educational data analyst generating objective, evidence-based 
reports on student engagement during classroom activities. 

CRITICAL CONSTRAINTS:
1. NEVER make identity inferences, personal judgments, or assumptions about 
   student characteristics beyond the provided numerical metrics.
2. NEVER use probabilistic language ("likely", "probably") for claims—only 
   state facts supported by thresholds (e.g., "gaze direction was >60° for 3min").
3. AVOID evaluative language; use neutral, descriptive terms.
4. FOCUS on behavioral metrics: gaze direction (°), posture stability (variance), 
   gesture frequency (events/min), task interaction indices.

OUTPUT FORMAT:
For each time interval and student:
- Metric Summary Table: [Time, Metric1, Metric2, ..., EngagementScore]
- Narrative: "During XX:YY-XX:ZZ, Student #N exhibited [metrics]. This correlates 
  with [feature-based explanation]. Recommended action: [teacher feedback]."

END CONSTRAINTS
```

#### Report Structure
```
CLASSROOM ENGAGEMENT ANALYSIS REPORT
Generated: 2025-10-29 14:32 UTC
Session: Period 3 (10:00-10:45)
Analyzer Version: v1.0 (YOLOv7 + DeepSORT + MediaPipe)

[EXECUTIVE SUMMARY]
- Session Duration: 45 min
- Students Tracked: 28 (IDs: 001-028)
- Mean Engagement: 62.3 ± 18.7 (std)
- Peak Engagement Interval: 10:15-10:20 (mean=78.5, math problem-solving task)

[DETAILED TIMELINE]
Time        Student#  Gaze(°)  Posture(σ)  Gesture(f/min)  Score  Narrative
10:00-10:03  S001     15.2     3.2         2.1             78     High focus on screen. Stable posture.
10:03-10:06  S001     42.8     8.5         1.5             52     Gaze drifting. Reduced interaction.
...

[STATISTICAL SUMMARY]
- Engagement distribution (histogram)
- Per-student trend analysis
- Correlation: gaze angle vs. task performance (if available)

[TEACHER RECOMMENDATIONS]
- Students with low engagement intervals: [list] → consider review/intervention
- High-engagement patterns: [describe] → potential peer mentoring targets

[METHODOLOGY NOTES]
- Pose model: MediaPipe Pose (v0.10.9)
- Feature definitions: See Appendix A
- Ethical constraints enforced: [✓ No identity inference] [✓ No probabilistic claims]
```

---

## Project Structure

```
videoanalysis/
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── config/
│   ├── config.yaml                          # System configuration (thresholds, weights)
│   └── gemini_system_prompt.txt              # Safety-constrained LLM prompt
├── src/
│   ├── __init__.py
│   ├── input_sync.py                        # Multi-modal input synchronization
│   ├── detector.py                          # YOLOv7 person detection
│   ├── tracker.py                           # DeepSORT multi-object tracking
│   ├── pose_estimator.py                    # MediaPipe/OpenPose interface
│   ├── feature_engineering.py               # Feature extraction & aggregation
│   ├── engagement_classifier.py             # Rule-based & ML classification
│   ├── report_generator.py                  # Gemini API integration
│   └── utils.py                             # Helper functions, visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb            # EDA on sample video
│   ├── 02_model_calibration.ipynb           # Threshold tuning
│   └── 03_report_generation.ipynb           # End-to-end pipeline demo
├── data/
│   ├── raw/                                 # Input video/audio files
│   ├── processed/                           # Synchronized, tracked data (HDF5)
│   └── annotations/                         # Manual labels for ML training
├── models/
│   ├── yolov7.pt                           # Pretrained YOLOv7 weights
│   └── engagement_classifier.pkl            # Trained XGBoost classifier
├── outputs/
│   ├── reports/                             # Generated engagement reports (HTML, PDF)
│   └── visualizations/                      # Debug videos, trajectory plots
└── tests/
    ├── test_detector.py
    ├── test_tracker.py
    └── test_feature_engineering.py
```

---

## Usage Guide

### Quick Start

#### 1. Installation
```bash
pip install -r requirements.txt
# For OpenPose: Follow https://github.com/CMU-Perceptual-Computing-Lab/openpose
```

#### 2. Configuration
Edit `config/config.yaml`:
```yaml
# Detection & Tracking
detection:
  confidence_threshold: 0.45
  nms_threshold: 0.5

tracking:
  max_age: 30  # Frames
  nn_budget: 100

# Feature Engineering
features:
  window_size: 3.0  # seconds
  window_overlap: 0.9
  gaze_threshold_engaged: 30  # degrees
  posture_variance_threshold: 5  # degrees

# Classification
classification:
  method: "rule_based"  # or "ml"
  weights:
    gaze: 0.4
    posture: 0.2
    gesture: 0.2
    interaction: 0.2

# Gemini API
gemini:
  model: "gemini-1.5-pro"
  api_key: "${GEMINI_API_KEY}"  # Load from environment
```

#### 3. Process Video
```python
from src.pipeline import EngagementAnalysisPipeline

pipeline = EngagementAnalysisPipeline(config_path="config/config.yaml")
results = pipeline.process_video(
    video_path="data/raw/classroom_session_001.mp4",
    output_dir="outputs/",
    audio_path="data/raw/audio_001.wav"  # Optional
)

# Generate report
report = pipeline.generate_report(results, include_visualizations=True)
report.save("outputs/reports/classroom_session_001_report.html")
```

---

## Academic Rigor & Validation

### Methodological Considerations

1. **Pose Estimation Validation**
   - Evaluate MediaPipe/OpenPose accuracy on labeled classroom data
   - Compute Percentage of Correct Keypoints (PCK) relative to manual annotation
   - Report per-joint confidence distributions

2. **Engagement Label Validity**
   - 3+ human raters independently label engagement (inter-rater agreement: Fleiss' κ)
   - Correlate automated scores with manual labels (Spearman's ρ)
   - Confusion matrix: precision, recall, F1-score per engagement category

3. **Temporal Consistency**
   - Analyze engagement score stability (frame-to-frame correlation)
   - Detect and report anomalous jumps (motion artifacts, tracking errors)

4. **Privacy & Ethics**
   - No facial recognition or identity matching
   - No sensitive attribute inference (e.g., emotion, socioeconomic status)
   - Data retention policy: Auto-delete individual frame data after 30 days; retain only aggregate metrics

5. **Reproducibility**
   - Version all model checkpoints & configurations
   - Log preprocessing steps (frame resize, normalization)
   - Publish results with error bars and confidence intervals

---

## Feature Interpretability

Each engagement score is **fully traceable** to underlying features:

Example Report Excerpt:
```
Student #5 | 10:12-10:15
├─ Gaze Angle: 22° (engaged, θ < 30°) → contribution: +0.32
├─ Posture Variance: 2.8° (stable) → contribution: +0.19
├─ Gesture Frequency: 3.2 events/min (moderate) → contribution: +0.15
├─ Interaction Index: 0.67 (tablet active) → contribution: +0.13
└─ TOTAL ENGAGEMENT SCORE: 79/100 [HIGH ENGAGEMENT]

Explanation: Student maintained forward-facing gaze toward the tablet display 
with stable upper-body posture. Gesture frequency suggests active problem-solving 
(writing, pointing). High interaction index confirms tablet-based task engagement.
```

---

## Troubleshooting & Limitations

### Known Limitations
1. **Occlusion**: Seated students partially obscured by desks reduce pose estimation accuracy
2. **Lighting**: Classroom lighting variations affect detection/tracking robustness
3. **Distance**: Students far from main camera may fail detection (resolve: multi-camera setup)
4. **Gesture ambiguity**: Pointing vs. writing gestures may be misclassified
5. **Multi-modal async**: Audio-video drift in unsynchronized recordings

### Mitigation Strategies
- Use overhead camera for torso-level capture (reduces occlusion)
- Calibrate lighting or use supplementary illumination
- Deploy multiple camera angles with cross-referencing
- Validate gesture classifiers on domain-specific dataset
- Implement robust timestamping (sync beeps, timecode)

---

## References & Further Reading

### Core Papers
1. **YOLOv7**: Wang et al., "YOLOv7: Trainable State-of-the-Art Object Detector" (2022)
   - https://arxiv.org/abs/2207.02696
2. **DeepSORT**: Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (2017)
   - https://arxiv.org/abs/1703.07402
3. **MediaPipe Pose**: Lugaresi et al., "MediaPipe: A Framework for Perceiving Streaming Media" (2019)
   - https://arxiv.org/abs/1906.08172

### Engagement & Learning Analytics
1. Csikszentmihalyi, M. (1990). "Flow: The Psychology of Optimal Experience"
2. Cocea, M., & Weibelzahl, S. (2009). "Disengagement Detection in Online Learning" (review)
3. Whitehill, J., et al. (2014). "The Facet Database of Facial Expressions and Emotions" (computer vision perspective)

---

## Contributing & Citation

For academic use, cite as:
```bibtex
@software{classroom_engagement_analyzer_2025,
  title={Classroom Engagement Analysis System: AI-Powered Student Activity Monitoring},
  author={[Your Name/Lab]},
  year={2025},
  url={https://github.com/your-org/videoanalysis}
}
```

---

## License
[Specify: MIT, Apache 2.0, CC-BY-4.0, etc.]

For questions or contributions, contact: [your-email]
