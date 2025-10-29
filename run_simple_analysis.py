# -*- coding: utf-8 -*-
"""
SIMPLE VIDEO ANALYSIS - Debugging & Quick Test
No complex dependencies - OpenCV only
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import json

print("=" * 80)
print("SIMPLE VIDEO ANALYSIS - Debug Version")
print("=" * 80)
print("\nThis script uses ONLY OpenCV (no YOLOv7, DeepSORT, or MediaPipe)")
print("Perfect for testing and quick results!\n")

# Check environment
print("[DEBUG] Environment Check:")
print(f"  OK OpenCV version: {cv2.__version__}")
print(f"  OK NumPy version: {np.__version__}")
print(f"  OK Python: {sys.version.split()[0]}")

# Find video file
video_file = "video_example.mp4"
if not Path(video_file).exists():
    print(f"\nX Error: {video_file} not found!")
    print(f"  Current directory: {Path.cwd()}")
    print(f"  Files: {list(Path.cwd().glob('*.mp4'))}")
    exit(1)

print(f"\n[DEBUG] Video File: {video_file}")
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print(f"X Failed to open video!")
    exit(1)

# Get video info
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.1f}sec")

# Simple analysis
print(f"\n[DEBUG] Analyzing first 50 frames...")
fgbg = cv2.createBackgroundSubtractorMOG2()

results = []
for frame_idx in range(min(50, total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Motion detection
        fgmask = fgbg.apply(frame)
        motion = np.sum(fgmask > 0) / fgmask.size * 100
        
        # Simple engagement score
        if motion > 5:
            score = 75
            level = "high"
        elif motion > 2:
            score = 50
            level = "moderate"
        else:
            score = 25
            level = "low"
        
        results.append({
            'frame': frame_idx,
            'motion': float(motion),
            'score': score,
            'level': level,
        })
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx+1:3d}: motion={motion:6.1f}%, score={score}")
    
    except Exception as e:
        print(f"  Frame {frame_idx}: Error - {e}")

cap.release()

# Results
print(f"\n[DEBUG] Analysis Complete!")
print(f"  Frames: {len(results)}")
if results:
    scores = [r['score'] for r in results]
    print(f"  Avg Score: {np.mean(scores):.1f}")
    print(f"  Min/Max: {np.min(scores)}/{np.max(scores)}")
    print(f"  Motion Mean: {np.mean([r['motion'] for r in results]):.1f}%")

# Save
output_dir = Path('outputs/reports')
output_dir.mkdir(parents=True, exist_ok=True)

json_file = output_dir / f'simple_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {json_file}")
print("\nOK SUCCESS - Script completed without errors!")
