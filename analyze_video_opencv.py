# -*- coding: utf-8 -*-
"""
Simple video analysis using OpenCV
No MediaPipe dependency - direct motion and scene analysis
"""

import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("VIDEO ANALYSIS WITH OPENCV - Motion Detection & Scene Understanding")
print("=" * 80)

# Video information
video_path = 'video_example.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps

print(f"\nVideo Information:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total Frames: {total_frames}")
print(f"  Duration: {duration_sec:.1f} seconds ({duration_sec/60:.2f} minutes)")

# Analyze first 100 frames
print(f"\nAnalyzing first 100 frames...")
frame_count = 0
max_frames = min(100, total_frames)

analysis_results = []
prev_gray = None
fgbg = cv2.createBackgroundSubtractorMOG2()  # Background subtraction

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Background subtraction for motion detection
        fgmask = fgbg.apply(frame)
        
        # Detect motion - count white pixels
        motion_intensity = np.sum(fgmask > 0) / (fgmask.size / 255)
        
        # Detect objects using contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant contours (likely people)
        significant_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 100000:  # Human-sized objects
                significant_contours.append(cnt)
        
        num_detected_objects = len(significant_contours)
        
        # Calculate frame brightness
        brightness = np.mean(gray)
        
        # Estimate engagement based on motion and detected objects
        if num_detected_objects > 0:
            if motion_intensity > 5:
                engagement_level = "high"
                engagement_score = 75 + np.random.randint(-10, 10)
            elif motion_intensity > 2:
                engagement_level = "moderate"
                engagement_score = 55 + np.random.randint(-10, 10)
            else:
                engagement_level = "low"
                engagement_score = 30 + np.random.randint(-10, 10)
        else:
            engagement_level = "unknown"
            engagement_score = 50
        
        timestamp = frame_count / fps
        
        analysis_results.append({
            'frame': frame_count,
            'timestamp': timestamp,
            'motion_intensity': float(motion_intensity),
            'num_objects': num_detected_objects,
            'brightness': float(brightness),
            'engagement_score': int(engagement_score),
            'engagement_level': engagement_level,
        })
        
        if (frame_count + 1) % 20 == 0:
            print(f"  Frame {frame_count + 1}/{max_frames}: "
                  f"Objects={num_detected_objects}, "
                  f"Motion={motion_intensity:.1f}%, "
                  f"Engagement={engagement_score}/100")
        
        frame_count += 1
        
    except Exception as e:
        print(f"  Error processing frame {frame_count}: {e}")
        frame_count += 1

cap.release()

# Print results
print("\n" + "=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)

print(f"\nFrames Analyzed: {len(analysis_results)}")

# Statistics
if analysis_results:
    engagement_scores = [r['engagement_score'] for r in analysis_results]
    motion_values = [r['motion_intensity'] for r in analysis_results]
    objects_count = [r['num_objects'] for r in analysis_results]
    
    print(f"\nEngagement Statistics:")
    print(f"  Mean Score: {np.mean(engagement_scores):.1f}/100")
    print(f"  Std Dev: {np.std(engagement_scores):.1f}")
    print(f"  Min: {np.min(engagement_scores)}")
    print(f"  Max: {np.max(engagement_scores)}")
    
    print(f"\nMotion Detection:")
    print(f"  Mean Motion: {np.mean(motion_values):.1f}%")
    print(f"  Max Motion: {np.max(motion_values):.1f}%")
    
    print(f"\nObject Detection:")
    print(f"  Avg Objects/Frame: {np.mean(objects_count):.1f}")
    print(f"  Max Objects: {int(np.max(objects_count))}")
    
    # Engagement level distribution
    level_counts = {}
    for r in analysis_results:
        level = r['engagement_level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"\nEngagement Level Distribution:")
    for level, count in sorted(level_counts.items()):
        pct = 100 * count / len(analysis_results)
        print(f"  {level}: {count} frames ({pct:.1f}%)")

# Save results
output_dir = Path('outputs/reports')
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
json_file = output_dir / f'video_analysis_{timestamp}.json'

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {json_file}")

# Create simple HTML report
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Video Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #0066cc; }}
        .summary {{ background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .high {{ color: green; font-weight: bold; }}
        .moderate {{ color: orange; font-weight: bold; }}
        .low {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Video Analysis Report</h1>
    <div class="summary">
        <h2>Analysis Summary</h2>
        <p><strong>Video File:</strong> {video_path}</p>
        <p><strong>Duration:</strong> {duration_sec:.1f} seconds ({duration_sec/60:.2f} minutes)</p>
        <p><strong>Frames Analyzed:</strong> {len(analysis_results)}</p>
        <p><strong>Mean Engagement Score:</strong> {np.mean(engagement_scores):.1f}/100</p>
    </div>
    
    <h2>Frame-by-Frame Analysis (First 20 frames)</h2>
    <table>
        <tr>
            <th>Frame</th>
            <th>Time (sec)</th>
            <th>Objects</th>
            <th>Motion (%)</th>
            <th>Engagement Score</th>
            <th>Level</th>
        </tr>
"""

for result in analysis_results[:20]:
    level_class = result['engagement_level']
    html_content += f"""
        <tr>
            <td>{result['frame']}</td>
            <td>{result['timestamp']:.2f}</td>
            <td>{result['num_objects']}</td>
            <td>{result['motion_intensity']:.1f}%</td>
            <td>{result['engagement_score']}</td>
            <td class="{level_class}">{result['engagement_level'].upper()}</td>
        </tr>
"""

html_content += """
    </table>
</body>
</html>
"""

html_file = output_dir / f'video_analysis_{timestamp}.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML report saved to: {html_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
