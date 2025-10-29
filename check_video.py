import cv2
import os

video_path = 'video_example.mp4'
if not os.path.exists(video_path):
    print('video_example.mp4를 찾을 수 없습니다')
    exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps

print('=' * 70)
print('VIDEO INFORMATION')
print('=' * 70)
print(f'File: {video_path}')
print(f'Resolution: {width}x{height}')
print(f'FPS: {fps}')
print(f'Total Frames: {total_frames}')
print(f'Duration: {duration_sec:.1f}sec ({duration_sec/60:.2f}min)')
print('=' * 70)

cap.release()
