# -*- coding: utf-8 -*-
"""
CLASSROOM ENGAGEMENT ANALYSIS SYSTEM - GOOGLE COLAB VERSION
============================================================

This script is designed to run on Google Colab for easy cloud-based video analysis.

SETUP INSTRUCTIONS FOR COLAB:
1. Run this cell in Google Colab
2. Upload video_example.mp4 to Colab (or mount Google Drive)
3. All dependencies will be installed automatically
4. Analysis results saved to outputs/reports/

FEATURES:
- Motion detection using background subtraction (MOG2)
- Object counting (estimated person detection)
- Engagement score estimation
- HTML report generation
- JSON data export

Author: Classroom Engagement Research Team
Date: 2025-10-29
Version: 1.0
"""

# ============================================================================
# SECTION 1: IMPORT LIBRARIES & SETUP
# ============================================================================

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Check if running on Colab
try:
    from google.colab import files
    IN_COLAB = True
    print("Running on Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally")

print("=" * 80)
print("CLASSROOM ENGAGEMENT ANALYSIS - VIDEO PROCESSING")
print("=" * 80)

# ============================================================================
# SECTION 2: SETUP FOR COLAB
# ============================================================================

if IN_COLAB:
    # Create working directory
    os.makedirs('outputs/reports', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    # Upload video file
    print("\nColab mode: Upload your video file")
    print("Click 'Choose Files' to upload video_example.mp4")
    uploaded = files.upload()
    
    # Get the uploaded filename
    if uploaded:
        video_path = list(uploaded.keys())[0]
        print(f"Video uploaded: {video_path}")
    else:
        print("No file uploaded. Using default path...")
        video_path = 'video_example.mp4'
else:
    video_path = 'video_example.mp4'

# ============================================================================
# SECTION 3: VIDEO INFORMATION EXTRACTION
# ============================================================================

def get_video_info(video_path: str) -> Dict:
    """
    Extract video file information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_sec': duration_sec,
    }

# Get video information
print(f"\nExtracting video information from: {video_path}")
try:
    video_info = get_video_info(video_path)
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']}")
    print(f"  Total Frames: {video_info['total_frames']}")
    print(f"  Duration: {video_info['duration_sec']:.1f} seconds ({video_info['duration_sec']/60:.2f} minutes)")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# ============================================================================
# SECTION 4: MOTION DETECTION & OBJECT COUNTING
# ============================================================================

def analyze_video(
    video_path: str,
    max_frames: int = 100,
    show_progress: bool = True
) -> List[Dict]:
    """
    Analyze video using motion detection and object counting.
    
    ALGORITHM:
    1. Background Subtraction (MOG2): Detects moving objects
    2. Contour Analysis: Identifies object boundaries
    3. Motion Intensity: Calculates % of moving pixels
    4. Engagement Scoring: Combines motion + object count
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to analyze (for speed)
        show_progress: Print progress messages
        
    Returns:
        List of frame analysis results
    """
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Limit frames for analysis
    max_frames = min(max_frames, total_frames)
    
    if show_progress:
        print(f"\nAnalyzing video (max {max_frames} frames)...")
    
    # Initialize background subtractor
    # MOG2 (Mixture of Gaussians v2) is robust to lighting changes
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    results = []
    frame_idx = 0
    
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # STEP 1: Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # STEP 2: Apply background subtraction
            # Output: fgmask where white = foreground (moving objects)
            fgmask = fgbg.apply(frame)
            
            # STEP 3: Calculate motion intensity
            # Motion = percentage of pixels showing movement
            motion_pixels = np.sum(fgmask > 0)
            total_pixels = fgmask.size
            motion_intensity = (motion_pixels / total_pixels) * 100
            
            # STEP 4: Find contours (object boundaries)
            contours, _ = cv2.findContours(
                fgmask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # STEP 5: Filter contours by size (human detection)
            # Typical person: 500-100000 pixels depending on distance
            significant_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter: avoid noise (too small) and shadows (too large)
                if 500 < area < 100000:
                    significant_contours.append(cnt)
            
            num_objects = len(significant_contours)
            
            # STEP 6: Calculate brightness (lighting condition indicator)
            brightness = np.mean(gray)
            
            # STEP 7: Estimate engagement score
            # High motion + multiple objects = high engagement
            if num_objects > 0:
                if motion_intensity > 5:  # High motion threshold
                    engagement_level = "high"
                    # Engagement score: 60-90 for high motion
                    engagement_score = int(70 + (motion_intensity % 20))
                elif motion_intensity > 2:  # Medium motion threshold
                    engagement_level = "moderate"
                    # Engagement score: 40-60 for medium motion
                    engagement_score = int(50 + (motion_intensity % 10))
                else:  # Low motion
                    engagement_level = "low"
                    # Engagement score: 20-40 for low motion
                    engagement_score = int(30 + (motion_intensity % 10))
            else:
                engagement_level = "unknown"
                engagement_score = 50  # Neutral score
            
            # Ensure score is in valid range [0-100]
            engagement_score = np.clip(engagement_score, 0, 100)
            
            # Calculate timestamp
            timestamp = frame_idx / fps
            
            # Store results for this frame
            results.append({
                'frame': frame_idx,
                'timestamp': timestamp,
                'motion_intensity': float(motion_intensity),
                'num_objects': num_objects,
                'brightness': float(brightness),
                'engagement_score': int(engagement_score),
                'engagement_level': engagement_level,
            })
            
            # Print progress
            if show_progress and (frame_idx + 1) % 20 == 0:
                pct = 100 * (frame_idx + 1) / max_frames
                print(f"  [{pct:5.1f}%] Frame {frame_idx + 1}/{max_frames}: "
                      f"Objects={num_objects:2d}, "
                      f"Motion={motion_intensity:6.1f}%, "
                      f"Engagement={engagement_score:3d}/100")
            
            frame_idx += 1
            
        except Exception as e:
            print(f"  Warning: Error processing frame {frame_idx}: {e}")
            frame_idx += 1
            continue
    
    cap.release()
    
    if show_progress:
        print(f"  Completed: {len(results)} frames analyzed")
    
    return results

# ============================================================================
# SECTION 5: RUN ANALYSIS
# ============================================================================

# Analyze video (first 100 frames for speed, increase for full analysis)
analysis_results = analyze_video(video_path, max_frames=100, show_progress=True)

# ============================================================================
# SECTION 6: COMPUTE STATISTICS
# ============================================================================

def compute_statistics(results: List[Dict]) -> Dict:
    """
    Compute statistical summary of analysis results.
    
    Args:
        results: List of frame analysis results
        
    Returns:
        Dictionary with statistical summary
    """
    
    if not results:
        return {}
    
    engagement_scores = [r['engagement_score'] for r in results]
    motion_values = [r['motion_intensity'] for r in results]
    objects_count = [r['num_objects'] for r in results]
    
    # Compute engagement level distribution
    level_counts = {}
    for r in results:
        level = r['engagement_level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    return {
        'num_frames': len(results),
        'engagement': {
            'mean': float(np.mean(engagement_scores)),
            'std': float(np.std(engagement_scores)),
            'min': int(np.min(engagement_scores)),
            'max': int(np.max(engagement_scores)),
        },
        'motion': {
            'mean_pct': float(np.mean(motion_values)),
            'max_pct': float(np.max(motion_values)),
        },
        'objects': {
            'avg_per_frame': float(np.mean(objects_count)),
            'max_detected': int(np.max(objects_count)),
        },
        'level_distribution': level_counts,
    }

# Compute statistics
stats = compute_statistics(analysis_results)

# ============================================================================
# SECTION 7: PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS RESULTS")
print("=" * 80)

print(f"\nFrames Analyzed: {stats['num_frames']}")

print(f"\nEngagement Statistics:")
print(f"  Mean Score: {stats['engagement']['mean']:.1f}/100")
print(f"  Std Dev: {stats['engagement']['std']:.1f}")
print(f"  Range: {stats['engagement']['min']}-{stats['engagement']['max']}")

print(f"\nMotion Detection:")
print(f"  Mean Motion: {stats['motion']['mean_pct']:.1f}%")
print(f"  Max Motion: {stats['motion']['max_pct']:.1f}%")

print(f"\nObject Detection:")
print(f"  Avg Objects/Frame: {stats['objects']['avg_per_frame']:.1f}")
print(f"  Max Objects: {stats['objects']['max_detected']}")

print(f"\nEngagement Level Distribution:")
for level, count in sorted(stats['level_distribution'].items()):
    pct = 100 * count / stats['num_frames']
    print(f"  {level:10s}: {count:3d} frames ({pct:5.1f}%)")

# ============================================================================
# SECTION 8: SAVE RESULTS
# ============================================================================

# Create output directory
output_dir = Path('outputs/reports')
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save as JSON (machine-readable)
json_file = output_dir / f'video_analysis_{timestamp}.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump({
        'metadata': {
            'video_file': video_path,
            'analysis_date': timestamp,
            'frames_analyzed': stats['num_frames'],
        },
        'statistics': stats,
        'frame_data': analysis_results[:20],  # First 20 frames for preview
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {json_file}")

# ============================================================================
# SECTION 9: GENERATE HTML REPORT
# ============================================================================

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Classroom Engagement Analysis Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 40px;
        }}
        h1 {{
            color: #333;
            border-bottom: 4px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #667eea;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }}
        .stat-box strong {{
            color: #333;
            display: block;
            margin-bottom: 5px;
        }}
        .stat-box .value {{
            font-size: 24px;
            color: #667eea;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .high {{ color: #28a745; font-weight: bold; }}
        .moderate {{ color: #ffc107; font-weight: bold; }}
        .low {{ color: #dc3545; font-weight: bold; }}
        .unknown {{ color: #6c757d; font-weight: bold; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Classroom Engagement Analysis Report</h1>
        
        <h2>Analysis Summary</h2>
        <div class="summary">
            <div class="stat-box">
                <strong>Video File</strong>
                <div class="value">{Path(video_path).name}</div>
            </div>
            <div class="stat-box">
                <strong>Duration</strong>
                <div class="value">{video_info['duration_sec']/60:.2f} min</div>
            </div>
            <div class="stat-box">
                <strong>Frames Analyzed</strong>
                <div class="value">{stats['num_frames']}</div>
            </div>
            <div class="stat-box">
                <strong>Avg Engagement</strong>
                <div class="value">{stats['engagement']['mean']:.0f}/100</div>
            </div>
        </div>
        
        <h2>Engagement Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Mean Score</td>
                <td>{stats['engagement']['mean']:.1f}/100</td>
                <td>Average engagement level across all frames</td>
            </tr>
            <tr>
                <td>Std Deviation</td>
                <td>{stats['engagement']['std']:.1f}</td>
                <td>Variability in engagement (higher = more unstable)</td>
            </tr>
            <tr>
                <td>Score Range</td>
                <td>{stats['engagement']['min']} - {stats['engagement']['max']}</td>
                <td>Minimum to maximum observed scores</td>
            </tr>
        </table>
        
        <h2>Engagement Level Distribution</h2>
        <table>
            <tr>
                <th>Level</th>
                <th>Frames</th>
                <th>Percentage</th>
                <th>Interpretation</th>
            </tr>
"""

for level, count in sorted(stats['level_distribution'].items(), key=lambda x: -x[1]):
    pct = 100 * count / stats['num_frames']
    class_name = level if level in ['high', 'moderate', 'low'] else 'unknown'
    interpretation = {
        'high': 'Students showing active engagement',
        'moderate': 'Students showing mixed engagement',
        'low': 'Students showing minimal engagement',
        'unknown': 'Unable to determine engagement level',
    }.get(level, '')
    
    html_content += f"""
            <tr>
                <td class="{class_name}">{level.upper()}</td>
                <td>{count}</td>
                <td>{pct:.1f}%</td>
                <td>{interpretation}</td>
            </tr>
"""

html_content += """
        </table>
        
        <h2>Motion & Object Detection</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Mean Motion</td>
                <td>""" + f"{stats['motion']['mean_pct']:.1f}%" + """</td>
                <td>Average percentage of moving pixels per frame</td>
            </tr>
            <tr>
                <td>Avg Objects</td>
                <td>""" + f"{stats['objects']['avg_per_frame']:.1f}" + """</td>
                <td>Average number of detected objects (students)</td>
            </tr>
            <tr>
                <td>Max Objects</td>
                <td>""" + f"{stats['objects']['max_detected']}" + """</td>
                <td>Peak number of students detected</td>
            </tr>
        </table>
        
        <div class="footer">
            <p>Report generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p>Classroom Engagement Analysis System v1.0 - OpenCV Motion Detection</p>
            <p>Analysis Method: Background Subtraction (MOG2) + Contour Analysis</p>
        </div>
    </div>
</body>
</html>
"""

html_file = output_dir / f'video_analysis_{timestamp}.html'
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML report saved to: {html_file}")

# ============================================================================
# SECTION 10: DOWNLOAD RESULTS (for Colab)
# ============================================================================

if IN_COLAB:
    print("\nColab: Preparing files for download...")
    print(f"Files ready to download in 'outputs/reports/'")
    print(f"  - {json_file.name}")
    print(f"  - {html_file.name}")
    
    # Optional: Download files
    print("\nTo download, run in next cell:")
    print("from google.colab import files")
    print(f"files.download('{json_file}')")
    print(f"files.download('{html_file}')")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
