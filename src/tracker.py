"""
Multi-Object Tracking Module using DeepSORT
Maintains persistent student IDs across video frames
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from deep_sort_realtime.deepsort import DeepSort

logger = logging.getLogger(__name__)


class DeepSORTTracker:
    """
    DeepSORT-based multi-object tracker for classroom students.
    
    Maintains persistent ID assignments across frames using:
    - Appearance features (CNN embeddings)
    - Kalman filtering (motion prediction)
    - Hungarian algorithm (frame-to-frame association)
    
    Attributes:
        tracker: DeepSort instance
        max_age: Max frames to keep undetected tracks
        min_hits: Min detections to confirm track
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        nn_budget: int = 100,
        max_distance: float = 0.2,
        iou_threshold: float = 0.3,
        use_gpu: bool = True,
    ):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Max frames to keep track alive without detection
            min_hits: Min detections before confirming track
            nn_budget: Memory budget for appearance features
            max_distance: Max Euclidean distance for matching (embedding space)
            iou_threshold: IOU threshold for bounding box matching
            use_gpu: Use GPU for appearance feature extraction
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            nn_budget=nn_budget,
            max_iou_distance=iou_threshold,
            max_euclidean_distance=max_distance,
            use_cuda=use_gpu,
        )
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.active_tracks = {}  # track_id -> metadata
        self.track_history = {}  # track_id -> list of (frame_idx, bbox, keypoints)
        
        logger.info(f"DeepSORT tracker initialized (max_age={max_age}, min_hits={min_hits})")
    
    def update(
        self,
        detections: List[Dict],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys:
                {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': float,
                    'features': np.ndarray (optional, shape (128,) or similar)
                }
            frame_idx: Current frame index (0-based)
            frame: Original frame (optional, used for feature extraction)
        
        Returns:
            List of tracked objects with keys:
            {
                'track_id': int,
                'student_id': str,  # "S{track_id:03d}"
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': float,
                'status': str,  # "confirmed" or "tentative"
            }
        """
        # Convert detections to DeepSort format
        bboxes_xywh = []
        confidences = []
        
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            w = x_max - x_min
            h = y_max - y_min
            cx = x_min + w / 2
            cy = y_min + h / 2
            
            bboxes_xywh.append([cx, cy, w, h])
            confidences.append(det['confidence'])
        
        bboxes_xywh = np.array(bboxes_xywh) if bboxes_xywh else np.empty((0, 4))
        confidences = np.array(confidences) if confidences else np.array([])
        
        # Update DeepSort tracker
        tracks = self.tracker.update_tracks(bboxes_xywh, confidences, frame)
        
        # Process confirmed tracks
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue  # Skip tentative tracks
            
            track_id = track.track_id
            x, y, w, h = track.to_ltrb()
            x_min, y_min, x_max, y_max = int(x), int(y), int(x + w), int(y + h)
            
            tracked_objects.append({
                'track_id': track_id,
                'student_id': f'S{track_id:03d}',
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': float(confidences[track.detection_idx]) 
                              if track.detection_idx >= 0 else 0.0,
                'status': 'confirmed',
            })
            
            # Update history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame_idx': frame_idx,
                'bbox': [x_min, y_min, x_max, y_max],
            })
        
        self.active_tracks = {t['track_id']: t for t in tracked_objects}
        logger.debug(f"Frame {frame_idx}: {len(tracked_objects)} confirmed tracks")
        
        return tracked_objects
    
    def get_active_track_ids(self) -> List[int]:
        """Get list of currently active track IDs."""
        return list(self.active_tracks.keys())
    
    def get_track_history(self, track_id: int) -> Optional[List[Dict]]:
        """
        Get history for a specific track.
        
        Args:
            track_id: Track ID to retrieve
        
        Returns:
            List of {frame_idx, bbox} dicts or None if track doesn't exist
        """
        return self.track_history.get(track_id)
    
    def get_statistics(self) -> Dict:
        """
        Get tracker statistics.
        
        Returns:
            {
                'num_active_tracks': int,
                'total_tracks_ever': int,
                'avg_track_length': float,
                'max_track_length': int,
            }
        """
        if not self.track_history:
            return {
                'num_active_tracks': 0,
                'total_tracks_ever': 0,
                'avg_track_length': 0.0,
                'max_track_length': 0,
            }
        
        track_lengths = [len(hist) for hist in self.track_history.values()]
        
        return {
            'num_active_tracks': len(self.active_tracks),
            'total_tracks_ever': len(self.track_history),
            'avg_track_length': np.mean(track_lengths),
            'max_track_length': np.max(track_lengths),
        }
    
    def reset(self):
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
        )
        self.active_tracks = {}
        self.track_history = {}
        logger.info("Tracker reset")


class TrackingResult:
    """Structured result from tracking a frame."""
    
    def __init__(
        self,
        frame_idx: int,
        timestamp: float,
        tracked_objects: List[Dict],
    ):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.tracked_objects = tracked_objects
    
    def get_track_ids(self) -> List[int]:
        """Get all track IDs in this frame."""
        return [obj['track_id'] for obj in self.tracked_objects]
    
    def get_by_track_id(self, track_id: int) -> Optional[Dict]:
        """Get tracking result for specific track ID."""
        for obj in self.tracked_objects:
            if obj['track_id'] == track_id:
                return obj
        return None
    
    def __repr__(self):
        return (f"TrackingResult(frame={self.frame_idx}, "
                f"timestamp={self.timestamp:.3f}, "
                f"num_tracks={len(self.tracked_objects)})")


def create_tracker(config: Dict) -> DeepSORTTracker:
    """
    Factory function to create tracker from config.
    
    Args:
        config: Configuration dictionary with tracking settings
    
    Returns:
        Initialized DeepSORTTracker
    """
    return DeepSORTTracker(
        max_age=config['tracking'].get('max_age', 30),
        min_hits=config['tracking'].get('min_hits', 3),
        nn_budget=config['tracking'].get('nn_budget', 100),
        max_distance=config['tracking'].get('max_distance', 0.2),
        iou_threshold=config['tracking'].get('iou_threshold', 0.3),
        use_gpu=config['tracking'].get('use_gpu', True),
    )
