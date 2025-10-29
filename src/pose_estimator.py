"""
Pose & Landmark Estimation Module
Supports MediaPipe Pose, Hand, and Face Landmarker
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")


class PoseEstimator:
    """
    Unified interface for pose estimation (MediaPipe Pose).
    
    Extracts:
    - 33 full-body landmarks (body)
    - 21 landmarks per hand (bilateral)
    - 468 face landmarks + head pose (optional)
    
    Each landmark: (x, y, z, confidence)
    """
    
    def __init__(
        self,
        backend: str = "mediapipe",
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        enable_hand_tracking: bool = True,
        enable_face_landmarks: bool = False,
    ):
        """
        Initialize pose estimator.
        
        Args:
            backend: "mediapipe" (others extensible)
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
            min_detection_confidence: Initial detection threshold
            min_tracking_confidence: Tracking confidence threshold
            static_image_mode: True for images, False for video
            enable_hand_tracking: Track hand landmarks
            enable_face_landmarks: Track face landmarks (privacy consideration)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Install: pip install mediapipe")
        
        self.backend = backend
        self.model_complexity = model_complexity
        self.enable_hand = enable_hand_tracking
        self.enable_face = enable_face_landmarks
        
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        if enable_hand_tracking:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=2,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self.hands = None
        
        if enable_face_landmarks:
            try:
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=min_detection_confidence,
                )
            except:
                logger.warning("Face detection not available")
                self.face_detector = None
        else:
            self.face_detector = None
        
        logger.info(f"PoseEstimator initialized (backend={backend}, "
                   f"complexity={model_complexity})")
    
    def estimate(self, frame: np.ndarray) -> Dict:
        """
        Estimate pose in frame.
        
        Args:
            frame: Input frame (HxWx3, BGR or RGB)
        
        Returns:
            {
                'pose': {
                    'landmarks': [(x, y, z, confidence), ...],  # 33 landmarks
                    'landmark_names': ['nose', 'left_eye', ...],
                    'visibility': [0.0-1.0, ...],
                },
                'hands': [
                    {
                        'landmarks': [(x, y, z, confidence), ...],  # 21 landmarks
                        'handedness': 'Left' or 'Right',
                    },
                    ...
                ],
                'face': {...} (optional)
            }
        """
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = {
            'pose': None,
            'hands': [],
            'face': None,
        }
        
        # Pose estimation
        pose_results = self.pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            landmarks = []
            for lm in pose_results.pose_landmarks.landmark:
                # Convert normalized coords to pixel coords
                x = lm.x * w
                y = lm.y * h
                z = lm.z * w  # Normalize z by width
                landmarks.append((x, y, z, lm.visibility))
            
            result['pose'] = {
                'landmarks': landmarks,
                'landmark_names': [lm.name for lm in self.mp_pose.PoseLandmark],
                'visibility': [lm.visibility for lm in pose_results.pose_landmarks.landmark],
            }
        
        # Hand detection
        if self.hands is not None:
            hand_results = self.hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                for hand_lm, handedness in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
                ):
                    landmarks = []
                    for lm in hand_lm.landmark:
                        x = lm.x * w
                        y = lm.y * h
                        z = lm.z * w
                        landmarks.append((x, y, z, 1.0))  # Hand landmarks always have confidence
                    
                    result['hands'].append({
                        'landmarks': landmarks,
                        'handedness': handedness.classification[0].label,
                    })
        
        return result
    
    def extract_gaze_vector(self, pose_data: Dict) -> Optional[Tuple[float, float, float]]:
        """
        Extract gaze direction from head landmarks.
        
        Returns:
            (nx, ny, nz) normalized head direction vector or None
        """
        if pose_data['pose'] is None:
            return None
        
        landmarks = pose_data['pose']['landmarks']
        
        # Keypoint indices in MediaPipe Pose
        P_NOSE = 0
        P_LEFT_EAR = 3
        P_RIGHT_EAR = 4
        
        try:
            nose = np.array(landmarks[P_NOSE][:3])
            left_ear = np.array(landmarks[P_LEFT_EAR][:3])
            right_ear = np.array(landmarks[P_RIGHT_EAR][:3])
            
            # Compute head normal
            ear_vec = right_ear - left_ear
            nose_to_mid_ear = nose - (left_ear + right_ear) / 2
            
            # Cross product for normal
            normal = np.cross(ear_vec, nose_to_mid_ear)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            
            return tuple(normal)
        except Exception as e:
            logger.debug(f"Failed to extract gaze vector: {e}")
            return None
    
    def extract_hand_positions(self, pose_data: Dict) -> List[Dict]:
        """
        Extract hand wrist positions and motion.
        
        Returns:
            List of {
                'handedness': 'Left'/'Right',
                'wrist': (x, y, z),
                'palm_center': (x, y, z),
            }
        """
        hands_info = []
        
        for hand in pose_data['hands']:
            landmarks = hand['landmarks']
            
            # Wrist is landmark 0
            wrist = np.array(landmarks[0][:3])
            
            # Palm center is average of key palm landmarks
            palm_landmarks = [0, 5, 9, 13, 17]  # Wrist + finger bases
            palm_center = np.mean([landmarks[i][:3] for i in palm_landmarks], axis=0)
            
            hands_info.append({
                'handedness': hand['handedness'],
                'wrist': tuple(wrist),
                'palm_center': tuple(palm_center),
            })
        
        return hands_info
    
    def extract_posture(self, pose_data: Dict) -> Optional[Dict]:
        """
        Extract posture metrics (shoulder-hip alignment, etc.).
        
        Returns:
            {
                'shoulder_left': (x, y, z),
                'shoulder_right': (x, y, z),
                'hip_left': (x, y, z),
                'hip_right': (x, y, z),
                'shoulder_hip_tilt_angle': float,  # degrees from vertical
            }
        """
        if pose_data['pose'] is None:
            return None
        
        landmarks = pose_data['pose']['landmarks']
        
        # Indices
        P_SHOULDER_LEFT = 11
        P_SHOULDER_RIGHT = 12
        P_HIP_LEFT = 23
        P_HIP_RIGHT = 24
        
        try:
            shoulder_left = np.array(landmarks[P_SHOULDER_LEFT][:3])
            shoulder_right = np.array(landmarks[P_SHOULDER_RIGHT][:3])
            hip_left = np.array(landmarks[P_HIP_LEFT][:3])
            hip_right = np.array(landmarks[P_HIP_RIGHT][:3])
            
            # Shoulder-hip axis
            shoulder_mid = (shoulder_left + shoulder_right) / 2
            hip_mid = (hip_left + hip_right) / 2
            posture_vec = hip_mid - shoulder_mid
            
            # Tilt angle from vertical
            vertical = np.array([0, 1, 0])
            cos_angle = np.dot(posture_vec, vertical) / (
                np.linalg.norm(posture_vec) * np.linalg.norm(vertical) + 1e-6
            )
            tilt_angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
            tilt_angle_deg = np.degrees(tilt_angle_rad)
            
            return {
                'shoulder_left': tuple(shoulder_left),
                'shoulder_right': tuple(shoulder_right),
                'hip_left': tuple(hip_left),
                'hip_right': tuple(hip_right),
                'shoulder_hip_tilt_angle_deg': float(tilt_angle_deg),
            }
        except Exception as e:
            logger.debug(f"Failed to extract posture: {e}")
            return None
    
    def close(self):
        """Clean up resources."""
        self.pose.close()
        if self.hands:
            self.hands.close()
        if self.face_detector:
            self.face_detector.close()
        logger.info("PoseEstimator closed")


def create_pose_estimator(config: Dict) -> PoseEstimator:
    """
    Factory function to create pose estimator from config.
    
    Args:
        config: Configuration dict with pose_estimation settings
    
    Returns:
        Initialized PoseEstimator
    """
    pose_config = config['pose_estimation']
    mediapipe_config = pose_config.get('mediapipe', {})
    
    return PoseEstimator(
        backend=pose_config.get('backend', 'mediapipe'),
        model_complexity=mediapipe_config.get('model_complexity', 1),
        min_detection_confidence=mediapipe_config.get('min_detection_confidence', 0.5),
        min_tracking_confidence=mediapipe_config.get('min_tracking_confidence', 0.5),
        static_image_mode=mediapipe_config.get('static_image_mode', False),
        enable_hand_tracking=mediapipe_config.get('hand', {}).get('enabled', True),
        enable_face_landmarks=mediapipe_config.get('face_landmarks', {}).get('enabled', False),
    )
