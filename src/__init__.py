"""
Classroom Engagement Analysis System
Academic-grade student engagement tracking with explainable AI
"""

__version__ = "1.0.0"
__author__ = "Your Lab/Organization"

from src.detector import PersonDetector
from src.tracker import DeepSORTTracker
from src.pose_estimator import PoseEstimator
from src.feature_engineering import FeatureExtractor
from src.engagement_classifier import EngagementClassifier
from src.report_generator import ReportGenerator

__all__ = [
    "PersonDetector",
    "DeepSORTTracker",
    "PoseEstimator",
    "FeatureExtractor",
    "EngagementClassifier",
    "ReportGenerator",
]
