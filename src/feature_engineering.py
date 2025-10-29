"""
Feature Engineering Module
Extracts quantitative engagement signals from pose, hand, and tracking data
Academic-grade implementation with full mathematical documentation
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EngagementFeatures:
    """
    Quantitative engagement signal features per student and time window.
    
    All features are normalized and interpretable for academic reporting.
    """
    
    def __init__(self, student_id: str, window_size_sec: float = 3.0):
        """
        Initialize feature accumulator for a student.
        
        Args:
            student_id: Student identifier (e.g., 'S001')
            window_size_sec: Sliding window duration (seconds)
        """
        self.student_id = student_id
        self.window_size_sec = window_size_sec
        
        # Feature buffers (deques for O(1) append/popleft)
        self.timestamps = deque()
        self.gaze_angles = deque()  # degrees from screen
        self.posture_tilts = deque()  # degrees from vertical
        self.hand_velocities = deque()  # pixels/frame
        self.gesture_events = deque()  # binary flags
        self.interaction_indices = deque()  # 0-1 tablet proximity
        
        # Aggregated features over window
        self.current_features = {}
    
    def add_frame(
        self,
        timestamp: float,
        gaze_angle_deg: Optional[float] = None,
        posture_tilt_deg: Optional[float] = None,
        hand_velocity_px: Optional[float] = None,
        is_gesture: bool = False,
        interaction_index: float = 0.0,
    ):
        """
        Add frame data to accumulator.
        
        Args:
            timestamp: Frame timestamp (seconds from video start)
            gaze_angle_deg: Head-to-screen angle (0° = facing screen, 90° = perpendicular)
            posture_tilt_deg: Shoulder-hip tilt from vertical (0° = upright)
            hand_velocity_px: Hand motion speed (pixels/frame)
            is_gesture: Whether frame contains pointing/writing gesture
            interaction_index: Tablet proximity score [0, 1]
        """
        # Add new data
        self.timestamps.append(timestamp)
        if gaze_angle_deg is not None:
            self.gaze_angles.append(gaze_angle_deg)
        if posture_tilt_deg is not None:
            self.posture_tilts.append(posture_tilt_deg)
        if hand_velocity_px is not None:
            self.hand_velocities.append(hand_velocity_px)
        self.gesture_events.append(1.0 if is_gesture else 0.0)
        self.interaction_indices.append(interaction_index)
        
        # Remove old data (older than window_size)
        while self.timestamps and (timestamp - self.timestamps[0]) > self.window_size_sec:
            self.timestamps.popleft()
            if len(self.gaze_angles) > 0:
                self.gaze_angles.popleft()
            if len(self.posture_tilts) > 0:
                self.posture_tilts.popleft()
            if len(self.hand_velocities) > 0:
                self.hand_velocities.popleft()
            self.gesture_events.popleft()
            self.interaction_indices.popleft()
    
    def compute_features(self) -> Dict:
        """
        Compute aggregated features over current window.
        
        Returns:
            {
                'timestamp': float,
                'gaze_mean_deg': float,
                'gaze_stability_variance_deg2': float,
                'posture_mean_tilt_deg': float,
                'posture_stability_variance_deg2': float,
                'hand_motion_mean_pxfs': float,
                'gesture_frequency_permin': float,
                'interaction_index_mean': float,
            }
        """
        features = {}
        
        if len(self.timestamps) == 0:
            return self._empty_features()
        
        features['timestamp'] = float(self.timestamps[-1])
        features['num_frames_in_window'] = len(self.timestamps)
        
        # Gaze features
        if len(self.gaze_angles) > 0:
            gaze_arr = np.array(list(self.gaze_angles))
            features['gaze_mean_deg'] = float(np.mean(gaze_arr))
            features['gaze_std_deg'] = float(np.std(gaze_arr))
            features['gaze_min_deg'] = float(np.min(gaze_arr))
            features['gaze_max_deg'] = float(np.max(gaze_arr))
        else:
            features['gaze_mean_deg'] = np.nan
            features['gaze_std_deg'] = np.nan
            features['gaze_min_deg'] = np.nan
            features['gaze_max_deg'] = np.nan
        
        # Posture features
        if len(self.posture_tilts) > 0:
            posture_arr = np.array(list(self.posture_tilts))
            features['posture_mean_tilt_deg'] = float(np.mean(posture_arr))
            features['posture_stability_std_deg'] = float(np.std(posture_arr))
        else:
            features['posture_mean_tilt_deg'] = np.nan
            features['posture_stability_std_deg'] = np.nan
        
        # Hand motion features
        if len(self.hand_velocities) > 0:
            hand_arr = np.array(list(self.hand_velocities))
            features['hand_motion_mean_pxfs'] = float(np.mean(hand_arr))
            features['hand_motion_max_pxfs'] = float(np.max(hand_arr))
            features['hand_motion_cumsum_px'] = float(np.sum(hand_arr))
        else:
            features['hand_motion_mean_pxfs'] = 0.0
            features['hand_motion_max_pxfs'] = 0.0
            features['hand_motion_cumsum_px'] = 0.0
        
        # Gesture frequency (normalized to per-minute rate)
        num_gestures = int(np.sum(self.gesture_events))
        window_min = self.window_size_sec / 60.0
        features['gesture_frequency_permin'] = num_gestures / max(window_min, 0.01)
        features['num_gestures_in_window'] = num_gestures
        
        # Interaction features
        if len(self.interaction_indices) > 0:
            interaction_arr = np.array(list(self.interaction_indices))
            features['interaction_mean_idx'] = float(np.mean(interaction_arr))
            features['interaction_max_idx'] = float(np.max(interaction_arr))
        else:
            features['interaction_mean_idx'] = 0.0
            features['interaction_max_idx'] = 0.0
        
        self.current_features = features
        return features
    
    def _empty_features(self) -> Dict:
        """Return default empty feature dict."""
        return {
            'timestamp': np.nan,
            'num_frames_in_window': 0,
            'gaze_mean_deg': np.nan,
            'gaze_std_deg': np.nan,
            'gaze_min_deg': np.nan,
            'gaze_max_deg': np.nan,
            'posture_mean_tilt_deg': np.nan,
            'posture_stability_std_deg': np.nan,
            'hand_motion_mean_pxfs': 0.0,
            'hand_motion_max_pxfs': 0.0,
            'hand_motion_cumsum_px': 0.0,
            'gesture_frequency_permin': 0.0,
            'num_gestures_in_window': 0,
            'interaction_mean_idx': 0.0,
            'interaction_max_idx': 0.0,
        }


class FeatureExtractor:
    """
    Main feature extraction pipeline.
    Processes pose/hand/tracking data → engagement features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dict with features settings
        """
        self.config = config
        self.feature_cfg = config['features']
        self.window_size = self.feature_cfg['window_size']
        self.window_overlap = self.feature_cfg['window_overlap']
        
        # Per-student feature accumulators
        self.student_features = {}  # student_id -> EngagementFeatures
        
        # Feature extraction state
        self.prev_hand_pos = {}  # track_id -> prev hand position
        self.fps = 30.0  # Default FPS, updated on first frame
        
        logger.info(f"FeatureExtractor initialized (window={self.window_size}s)")
    
    def extract_per_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        tracked_objects: List[Dict],
        pose_results: Dict,
        fps: float = 30.0,
    ) -> Dict[str, Dict]:
        """
        Extract features from single frame data.
        
        Args:
            frame_idx: Frame index (0-based)
            timestamp_sec: Timestamp in seconds
            tracked_objects: List of tracked student objects
            pose_results: Dict of pose estimation results per track_id
            fps: Video framerate (for normalization)
        
        Returns:
            {
                'S001': {aggregated features over window},
                'S002': {...},
                ...
            }
        """
        self.fps = fps
        frame_features = {}
        
        for tracked_obj in tracked_objects:
            track_id = tracked_obj['track_id']
            student_id = tracked_obj['student_id']
            bbox = tracked_obj['bbox']
            
            # Initialize student accumulator if needed
            if student_id not in self.student_features:
                self.student_features[student_id] = EngagementFeatures(
                    student_id, self.window_size
                )
            
            # Extract frame-level features
            gaze_angle = self._compute_gaze_angle(track_id, pose_results)
            posture_tilt = self._compute_posture_tilt(track_id, pose_results)
            hand_velocity = self._compute_hand_velocity(track_id, pose_results, fps)
            is_gesture = self._detect_gesture(track_id, pose_results)
            interaction_idx = self._compute_interaction_index(track_id, pose_results, bbox)
            
            # Add to accumulator
            accumulator = self.student_features[student_id]
            accumulator.add_frame(
                timestamp=timestamp_sec,
                gaze_angle_deg=gaze_angle,
                posture_tilt_deg=posture_tilt,
                hand_velocity_px=hand_velocity,
                is_gesture=is_gesture,
                interaction_index=interaction_idx,
            )
            
            # Compute aggregated features
            frame_features[student_id] = accumulator.compute_features()
        
        return frame_features
    
    def _compute_gaze_angle(
        self,
        track_id: int,
        pose_results: Dict,
    ) -> Optional[float]:
        """
        Compute head-to-screen gaze angle.
        
        Returns:
            Angle in degrees (0° = facing screen, 90° = perpendicular)
        """
        if track_id not in pose_results:
            return None
        
        pose_data = pose_results[track_id]
        if pose_data.get('pose') is None:
            return None
        
        landmarks = pose_data['pose']['landmarks']
        
        # MediaPipe indices
        P_NOSE = 0
        P_LEFT_EYE = 1
        P_RIGHT_EYE = 2
        
        try:
            nose = np.array(landmarks[P_NOSE][:3])
            left_eye = np.array(landmarks[P_LEFT_EYE][:3])
            right_eye = np.array(landmarks[P_RIGHT_EYE][:3])
            
            # Gaze vector (nose-to-eye midpoint)
            eye_mid = (left_eye + right_eye) / 2
            gaze_vec = eye_mid - nose
            gaze_vec = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-6)
            
            # Screen normal (assumption: screen is orthogonal to image plane)
            screen_normal = np.array([0, 0, -1])  # Into screen
            
            # Angle between gaze and screen
            cos_angle = np.dot(gaze_vec, screen_normal)
            angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
            angle_deg = np.degrees(angle_rad)
            
            return float(angle_deg)
        except Exception as e:
            logger.debug(f"Failed to compute gaze angle: {e}")
            return None
    
    def _compute_posture_tilt(
        self,
        track_id: int,
        pose_results: Dict,
    ) -> Optional[float]:
        """
        Compute upper-body posture tilt angle.
        
        Returns:
            Tilt in degrees (0° = upright, 45° = moderate lean, 90° = horizontal)
        """
        if track_id not in pose_results:
            return None
        
        pose_data = pose_results[track_id]
        posture_metrics = pose_data.get('posture')
        
        if posture_metrics is None:
            return None
        
        return posture_metrics.get('shoulder_hip_tilt_angle_deg')
    
    def _compute_hand_velocity(
        self,
        track_id: int,
        pose_results: Dict,
        fps: float,
    ) -> float:
        """
        Compute hand motion velocity.
        
        Returns:
            Pixels per frame
        """
        if track_id not in pose_results:
            return 0.0
        
        pose_data = pose_results[track_id]
        hands = pose_data.get('hands', [])
        
        if not hands:
            self.prev_hand_pos[track_id] = None
            return 0.0
        
        # Use first hand (left or right)
        current_hand = hands[0]
        current_pos = np.array(current_hand['wrist'][:2])
        
        if track_id in self.prev_hand_pos and self.prev_hand_pos[track_id] is not None:
            prev_pos = self.prev_hand_pos[track_id]
            velocity = np.linalg.norm(current_pos - prev_pos)
        else:
            velocity = 0.0
        
        self.prev_hand_pos[track_id] = current_pos
        return float(velocity)
    
    def _detect_gesture(
        self,
        track_id: int,
        pose_results: Dict,
    ) -> bool:
        """
        Detect pointing/writing gesture.
        
        Returns:
            True if gesture detected
        """
        if track_id not in pose_results:
            return False
        
        pose_data = pose_results[track_id]
        hands = pose_data.get('hands', [])
        
        # Gesture heuristic: hand raised (hand wrist above shoulder)
        if not hands or pose_data.get('pose') is None:
            return False
        
        try:
            landmarks = pose_data['pose']['landmarks']
            P_SHOULDER_RIGHT = 12
            shoulder_y = landmarks[P_SHOULDER_RIGHT][1]
            
            for hand in hands:
                hand_wrist_y = hand['wrist'][1]
                # Hand raised if wrist is above shoulder
                if hand_wrist_y < shoulder_y - 20:  # 20px margin
                    return True
            
            return False
        except Exception as e:
            logger.debug(f"Failed to detect gesture: {e}")
            return False
    
    def _compute_interaction_index(
        self,
        track_id: int,
        pose_results: Dict,
        bbox: List[int],
    ) -> float:
        """
        Compute tablet interaction score.
        
        Returns:
            Score in [0, 1] (0 = no interaction, 1 = hands on tablet)
        """
        if track_id not in pose_results:
            return 0.0
        
        pose_data = pose_results[track_id]
        hands = pose_data.get('hands', [])
        
        if not hands:
            return 0.0
        
        # Heuristic: hands within bbox and close to screen plane
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        tablet_region = [
            x_min + bbox_width * 0.1,
            y_min + bbox_height * 0.3,
            x_max - bbox_width * 0.1,
            y_max - bbox_height * 0.1,
        ]
        
        interaction_score = 0.0
        for hand in hands:
            wrist_x, wrist_y = hand['wrist'][:2]
            
            # Check if wrist in tablet region
            if (tablet_region[0] <= wrist_x <= tablet_region[2] and
                tablet_region[1] <= wrist_y <= tablet_region[3]):
                interaction_score = max(interaction_score, 0.8)
        
        return float(interaction_score)


def create_feature_extractor(config: Dict) -> FeatureExtractor:
    """Factory function."""
    return FeatureExtractor(config)
