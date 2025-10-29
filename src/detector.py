"""
Person Detection Module using YOLOv7
Extracts person bounding boxes from video frames
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    YOLOv7-based person detector for classroom environments.
    
    Attributes:
        model: Loaded YOLOv7 model
        device: torch device (cuda/cpu)
        confidence_threshold: Minimum confidence for detection (0.0-1.0)
        nms_threshold: NMS threshold for duplicate removal
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.5,
        device: str = "cuda",
        half_precision: bool = True,
    ):
        """
        Initialize YOLOv7 detector.
        
        Args:
            model_path: Path to pretrained YOLOv7 weights (.pt file)
            confidence_threshold: Minimum detection confidence (0.45 recommended)
            nms_threshold: Non-Maximum Suppression threshold (0.5 typical)
            device: "cuda" or "cpu"
            half_precision: Use FP16 for faster inference
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        try:
            # Try loading from torch
            self.model = torch.hub.load('ultralytics/yolov7', 'custom', 
                                       path=model_path, force_reload=False)
        except Exception as e:
            logger.warning(f"Failed to load model from hub: {e}. Using local import...")
            # Fallback: Import locally if torch.hub fails
            import sys
            sys.path.append('models')
            from models import yolov7
            self.model = yolov7.DetectMultiBackend(model_path, device=self.device)
        
        self.model.to(self.device)
        self.model.eval()
        
        if half_precision and str(self.device) != 'cpu':
            self.model.half()
        
        self.half = half_precision and str(self.device) != 'cpu'
        logger.info(f"YOLOv7 detector initialized on {self.device}")
    
    def detect(
        self, 
        frame: np.ndarray,
        target_size: Tuple[int, int] = (640, 480),
        classes: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input frame (HxWx3, RGB or BGR)
            target_size: Resize frame to this size (width, height)
            classes: Filter by class IDs (default: [0] for person in COCO)
        
        Returns:
            List of detections:
            [{
                'bbox': [x_min, y_min, x_max, y_max],  # Absolute pixel coordinates
                'confidence': float,
                'class': int,
                'class_name': str,
            }, ...]
        
        Raises:
            ValueError: If frame is invalid
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
        
        if classes is None:
            classes = [0]  # COCO person class
        
        h, w = frame.shape[:2]
        
        # Run inference
        with torch.no_grad():
            results = self.model(frame, size=target_size[0])
        
        detections = []
        preds = results.pred[0]  # Predictions tensor
        
        # Parse predictions
        for *xyxy, conf, cls in reversed(preds):
            cls = int(cls)
            if cls not in classes:
                continue
            
            conf = float(conf)
            if conf < self.confidence_threshold:
                continue
            
            # Convert to absolute pixel coordinates
            x_min, y_min, x_max, y_max = map(lambda x: int(x.item()), xyxy)
            
            # Clamp to frame boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
            
            detections.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'confidence': conf,
                'class': cls,
                'class_name': 'person',
            })
        
        logger.debug(f"Detected {len(detections)} persons in frame")
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        target_size: Tuple[int, int] = (640, 480),
    ) -> List[List[Dict]]:
        """
        Batch detection for multiple frames.
        
        Args:
            frames: List of frames
            target_size: Target inference size
        
        Returns:
            List of detection lists (one per frame)
        """
        batch_results = []
        for frame in frames:
            detections = self.detect(frame, target_size)
            batch_results.append(detections)
        return batch_results
    
    def get_cropped_persons(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        padding: float = 0.1,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract cropped images of detected persons.
        
        Args:
            frame: Original frame
            detections: List of detection dictionaries
            padding: Extra padding around bbox as fraction (0.1 = 10%)
        
        Returns:
            List of (cropped_image, detection_metadata) tuples
        """
        cropped_persons = []
        h, w = frame.shape[:2]
        
        for det in detections:
            x_min, y_min, x_max, y_max = det['bbox']
            
            # Add padding
            pad_x = int((x_max - x_min) * padding)
            pad_y = int((y_max - y_min) * padding)
            
            x_min_p = max(0, x_min - pad_x)
            y_min_p = max(0, y_min - pad_y)
            x_max_p = min(w, x_max + pad_x)
            y_max_p = min(h, y_max + pad_y)
            
            cropped = frame[y_min_p:y_max_p, x_min_p:x_max_p]
            cropped_persons.append((cropped, det))
        
        return cropped_persons


def create_detector(config: Dict) -> PersonDetector:
    """
    Factory function to create detector from config.
    
    Args:
        config: Configuration dictionary with detection settings
    
    Returns:
        Initialized PersonDetector
    """
    return PersonDetector(
        model_path=config['detection'].get('model_path', 'models/yolov7.pt'),
        confidence_threshold=config['detection'].get('confidence_threshold', 0.45),
        nms_threshold=config['detection'].get('nms_threshold', 0.5),
        device=config['detection'].get('device', 'cuda'),
        half_precision=config['detection'].get('half_precision', True),
    )
