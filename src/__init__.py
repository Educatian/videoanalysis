"""
Classroom Engagement Analysis System
Academic-grade student engagement tracking with explainable AI

NOTE: This module uses lazy loading for optional dependencies.
Some components may not be available if packages are not installed.
Use standalone scripts for simpler workflows without dependencies.
"""

__version__ = "1.0.0"
__author__ = "Classroom Engagement Research Team"

# Try importing components, skip if dependencies missing
def _safe_import(module_name, attr_name):
    """Safely import a module attribute, return None if not available."""
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import {attr_name} from {module_name}: {e}")
        return None

# Lazy imports with fallback
PersonDetector = _safe_import('src.detector', 'PersonDetector')
DeepSORTTracker = _safe_import('src.tracker', 'DeepSORTTracker')
PoseEstimator = _safe_import('src.pose_estimator', 'PoseEstimator')
FeatureExtractor = _safe_import('src.feature_engineering', 'FeatureExtractor')
EngagementClassifier = _safe_import('src.engagement_classifier', 'EngagementClassifier')
ReportGenerator = _safe_import('src.report_generator', 'ReportGenerator')

# List available components
__all__ = [
    "PersonDetector",
    "DeepSORTTracker",
    "PoseEstimator",
    "FeatureExtractor",
    "EngagementClassifier",
    "ReportGenerator",
]

print("""
╔════════════════════════════════════════════════════════════════════╗
║   Classroom Engagement Analysis System - Module Loader v1.0       ║
╚════════════════════════════════════════════════════════════════════╝

✓ Available components:
""")

available = [name for name in __all__ if globals()[name] is not None]
unavailable = [name for name in __all__ if globals()[name] is None]

for component in available:
    print(f"  ✓ {component}")

if unavailable:
    print(f"\n⚠ Unavailable components (missing dependencies):")
    for component in unavailable:
        print(f"  ✗ {component}")

print(f"""
For standalone video analysis without dependencies, use:
  • python analyze_video_opencv.py (OpenCV only)
  • python classroom_engagement_colab.py (Google Colab)

For full features, install all dependencies:
  pip install -r requirements.txt
""")
