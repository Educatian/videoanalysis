"""
Engagement Classification Module
Rule-based scoring with full interpretability for academic reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EngagementClassifier:
    """
    Classify engagement level from extracted features.
    
    Uses interpretable, rule-based scoring for academic transparency.
    All scores and decisions fully traceable to specific feature values.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize classifier.
        
        Args:
            config: Configuration dict with classification settings
        """
        self.config = config
        self.class_cfg = config['classification']
        self.method = self.class_cfg.get('method', 'rule_based')
        
        if self.method == 'rule_based':
            self.rule_cfg = self.class_cfg.get('rule_based', {})
            self.weights = self.rule_cfg.get('weights', {
                'gaze': 0.40,
                'posture': 0.20,
                'gesture': 0.20,
                'interaction': 0.20,
            })
            
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
            
            # Feature thresholds
            self.gaze_engaged_threshold = config['features']['gaze']['eye_contact_threshold_deg']
            self.posture_stable_threshold = config['features']['posture']['stability_variance_threshold_deg']
            
            logger.info(f"EngagementClassifier (rule_based) initialized with weights: {self.weights}")
        
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
    
    def classify(self, features: Dict) -> Dict:
        """
        Classify engagement from feature dict.
        
        Args:
            features: Feature dict from FeatureExtractor, containing:
            {
                'gaze_mean_deg': float,
                'gaze_std_deg': float,
                'posture_mean_tilt_deg': float,
                'posture_stability_std_deg': float,
                'hand_motion_mean_pxfs': float,
                'gesture_frequency_permin': float,
                'interaction_mean_idx': float,
            }
        
        Returns:
            {
                'engagement_score': float (0-100),
                'engagement_level': str ('disengaged', 'passive', 'engaged'),
                'component_scores': {
                    'gaze_score': float (0-100),
                    'posture_score': float (0-100),
                    'gesture_score': float (0-100),
                    'interaction_score': float (0-100),
                },
                'feature_values': {...},
                'explanation': str,  # Human-readable interpretation
            }
        """
        if self.method == 'rule_based':
            return self._classify_rule_based(features)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
    
    def _classify_rule_based(self, features: Dict) -> Dict:
        """
        Rule-based classification with transparent scoring.
        
        Mathematical formulation:
        EngagementScore = Σ(w_i * f_i(x_i))
        
        Where:
        - w_i: feature weight (normalized)
        - f_i(x_i): feature scoring function
        """
        result = {
            'component_scores': {},
            'feature_values': features,
            'debug_info': {},
        }
        
        # Component scores (each 0-100)
        gaze_score = self._score_gaze(features)
        posture_score = self._score_posture(features)
        gesture_score = self._score_gesture(features)
        interaction_score = self._score_interaction(features)
        
        result['component_scores'] = {
            'gaze_score': float(gaze_score),
            'posture_score': float(posture_score),
            'gesture_score': float(gesture_score),
            'interaction_score': float(interaction_score),
        }
        
        # Weighted sum (0-100)
        engagement_score = (
            self.weights['gaze'] * gaze_score +
            self.weights['posture'] * posture_score +
            self.weights['gesture'] * gesture_score +
            self.weights['interaction'] * interaction_score
        )
        
        result['engagement_score'] = float(np.clip(engagement_score, 0, 100))
        
        # Classification
        levels = self.rule_cfg.get('engagement_levels', {
            'disengaged': [0, 33],
            'passive': [34, 66],
            'engaged': [67, 100],
        })
        
        for level_name, [lower, upper] in levels.items():
            if lower <= result['engagement_score'] <= upper:
                result['engagement_level'] = level_name
                break
        else:
            result['engagement_level'] = 'unknown'
        
        # Generate explanation
        result['explanation'] = self._generate_explanation(
            result, features, result['component_scores']
        )
        
        return result
    
    def _score_gaze(self, features: Dict) -> float:
        """
        Score gaze direction engagement.
        
        Formula: f_gaze(θ) = max(0, 100 * (1 - θ/90))
        
        Interpretation:
        - θ < 30°: high engagement (facing screen)
        - 30° ≤ θ < 60°: moderate
        - θ ≥ 60°: low engagement (looking away)
        """
        gaze_angle = features.get('gaze_mean_deg')
        
        if gaze_angle is None or np.isnan(gaze_angle):
            return 50.0  # Neutral score if missing
        
        # Linear decay: 0° → 100, 90° → 0
        score = max(0, 100 * (1 - gaze_angle / 90.0))
        return float(np.clip(score, 0, 100))
    
    def _score_posture(self, features: Dict) -> float:
        """
        Score posture stability.
        
        Formula: f_posture(σ) = 100 * exp(-(σ/σ_0)^2)
        
        Where σ_0 is stability threshold (e.g., 5°)
        
        Interpretation:
        - Low variance (σ < 5°): stable, engaged posture
        - High variance (σ > 15°): fidgeting, disengaged
        """
        posture_variance = features.get('posture_stability_std_deg')
        
        if posture_variance is None or np.isnan(posture_variance):
            return 50.0
        
        # Gaussian penalty
        sigma_0 = self.posture_stable_threshold
        exponent = (posture_variance / sigma_0) ** 2
        score = 100 * np.exp(-exponent)
        
        return float(np.clip(score, 0, 100))
    
    def _score_gesture(self, features: Dict) -> float:
        """
        Score hand gesture activity.
        
        Formula: f_gesture(f) = 100 * min(1, f / f_max)
        
        Where f_max is expected gesture rate (e.g., 5 gestures/min)
        
        Interpretation:
        - No gestures: 0 (disengaged or passive listening)
        - Frequent gestures: 100 (active engagement)
        """
        gesture_freq = features.get('gesture_frequency_permin', 0.0)
        
        # Saturating function: 0 at f=0, plateaus at f≥5
        max_freq = 5.0
        score = min(100, 100 * gesture_freq / max_freq)
        
        return float(np.clip(score, 0, 100))
    
    def _score_interaction(self, features: Dict) -> float:
        """
        Score tablet/device interaction.
        
        Formula: f_interact(I) = 100 * I_mean
        
        Where I is normalized interaction index [0, 1]
        
        Interpretation:
        - I = 0: no device interaction
        - I = 1: active hands-on-device engagement
        """
        interaction_idx = features.get('interaction_mean_idx', 0.0)
        
        score = 100 * interaction_idx
        return float(np.clip(score, 0, 100))
    
    def _generate_explanation(
        self,
        result: Dict,
        features: Dict,
        component_scores: Dict,
    ) -> str:
        """
        Generate human-readable explanation of classification.
        
        Academic-grade: Facts based on thresholds, no speculation.
        """
        parts = []
        
        # Engagement level summary
        engagement_level = result['engagement_level']
        engagement_score = result['engagement_score']
        parts.append(
            f"Engagement Score: {engagement_score:.1f}/100 [{engagement_level.upper()}]"
        )
        
        # Gaze analysis
        gaze_angle = features.get('gaze_mean_deg')
        gaze_score = component_scores['gaze_score']
        if gaze_angle is not None:
            if gaze_angle < 30:
                gaze_status = "HIGH (facing screen)"
            elif gaze_angle < 60:
                gaze_status = "MODERATE (partial attention)"
            else:
                gaze_status = "LOW (looking away)"
            parts.append(
                f"• Gaze: {gaze_angle:.1f}° from screen → {gaze_status} "
                f"(score: {gaze_score:.1f})"
            )
        
        # Posture analysis
        posture_var = features.get('posture_stability_std_deg')
        posture_score = component_scores['posture_score']
        if posture_var is not None:
            if posture_var < 5:
                posture_status = "STABLE (upright seated)"
            elif posture_var < 15:
                posture_status = "MODERATE (some movement)"
            else:
                posture_status = "UNSTABLE (fidgeting)"
            parts.append(
                f"• Posture Stability: σ={posture_var:.1f}° → {posture_status} "
                f"(score: {posture_score:.1f})"
            )
        
        # Gesture analysis
        gesture_freq = features.get('gesture_frequency_permin', 0.0)
        gesture_score = component_scores['gesture_score']
        if gesture_freq > 0:
            parts.append(
                f"• Hand Gestures: {gesture_freq:.1f} events/min → ACTIVE "
                f"(score: {gesture_score:.1f})"
            )
        else:
            parts.append(
                f"• Hand Gestures: None detected → PASSIVE "
                f"(score: {gesture_score:.1f})"
            )
        
        # Interaction analysis
        interaction_idx = features.get('interaction_mean_idx', 0.0)
        interaction_score = component_scores['interaction_score']
        if interaction_idx > 0.5:
            interaction_status = "ACTIVE (hands on device)"
        elif interaction_idx > 0:
            interaction_status = "PARTIAL (nearby device)"
        else:
            interaction_status = "ABSENT (no device interaction)"
        parts.append(
            f"• Device Interaction: {interaction_idx:.2f} → {interaction_status} "
            f"(score: {interaction_score:.1f})"
        )
        
        # Composite interpretation
        parts.append("\n[Interpretation]")
        if engagement_score >= 67:
            parts.append(
                "Student exhibits indicators of high engagement: sustained attention, "
                "active participation, and device interaction."
            )
        elif engagement_score >= 34:
            parts.append(
                "Student exhibits moderate engagement: mixed signals from attention, "
                "posture, and activity levels. May benefit from re-engagement strategies."
            )
        else:
            parts.append(
                "Student exhibits low engagement indicators: minimal attention, "
                "posture instability, or reduced activity. Intervention recommended."
            )
        
        return "\n".join(parts)
    
    def batch_classify(self, features_list: List[Dict]) -> List[Dict]:
        """
        Classify multiple feature sets (e.g., per-frame results).
        
        Args:
            features_list: List of feature dicts
        
        Returns:
            List of classification results
        """
        return [self.classify(features) for features in features_list]


def create_classifier(config: Dict) -> EngagementClassifier:
    """Factory function."""
    return EngagementClassifier(config)
