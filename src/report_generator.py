"""
Report Generation Module
Gemini API integration for evidence-based engagement narratives with ethical constraints
"""

import os
import json
from typing import Dict, List, Optional
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google.generativeai not available")


class ReportGenerator:
    """
    Generate academically rigorous engagement reports using Gemini API.
    
    Safety & Ethics:
    - No identity inference or personal judgments
    - No probabilistic claims without data thresholds
    - Evidence-based only (metrics → conclusions)
    - Privacy-compliant narrative generation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dict with report_generation settings
        """
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini API not available. Install: pip install google-generativeai")
        
        self.config = config
        self.report_cfg = config['report_generation']
        
        # Initialize Gemini
        if self.report_cfg.get('enabled', True) and GEMINI_AVAILABLE:
            api_key = os.getenv(
                self.report_cfg['gemini'].get('api_key_env', 'GEMINI_API_KEY')
            )
            if api_key:
                genai.configure(api_key=api_key)
                self.model_name = self.report_cfg['gemini'].get('model', 'gemini-1.5-pro')
                self.model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        'temperature': self.report_cfg['gemini'].get('temperature', 0.3),
                        'top_p': self.report_cfg['gemini'].get('top_p', 0.95),
                        'top_k': self.report_cfg['gemini'].get('top_k', 40),
                        'max_output_tokens': self.report_cfg['gemini'].get('max_output_tokens', 4096),
                    },
                    system_prompt=self._load_system_prompt(),
                )
                logger.info(f"Gemini model {self.model_name} initialized")
            else:
                logger.warning("GEMINI_API_KEY not found in environment")
                self.model = None
        else:
            self.model = None
    
    def _load_system_prompt(self) -> str:
        """Load safety-constrained system prompt for Gemini."""
        # Try loading from config
        system_prompt = self.report_cfg.get('system_prompt', '')
        
        if not system_prompt:
            # Fallback to default
            system_prompt = """
You are an objective educational data analyst generating evidence-based reports 
on student engagement during classroom activities.

CRITICAL CONSTRAINTS:
1. NEVER make identity inferences, personal judgments, or assumptions about 
   student characteristics beyond the provided numerical metrics.
2. NEVER use probabilistic language ("likely", "probably", "seems") for claims.
   Only state facts supported by measured thresholds (e.g., 
   "gaze direction was >60° from screen center for 3 minutes").
3. AVOID evaluative language; use neutral, descriptive terms.
4. FOCUS on behavioral metrics: gaze angle (°), posture stability (variance), 
   gesture frequency (events/min), task interaction indices.
5. When explaining patterns, cite specific metric values and thresholds.
6. Format numerical information clearly with units and precision.

OUTPUT INSTRUCTIONS:
- For each time interval and student, provide metrics and factual interpretation.
- Separate measured observations from pedagogical recommendations.
- Do NOT infer psychological states, emotions, or personality traits.

PROHIBITED OUTPUTS:
- No personality assessments
- No ability or learning potential judgments
- No emotional state inferences
- No socioeconomic or demographic predictions
- No medical/psychological diagnoses
"""
        
        return system_prompt
    
    def generate_student_narrative(
        self,
        student_id: str,
        timeblock: str,
        features: Dict,
        classification: Dict,
    ) -> str:
        """
        Generate narrative for single student-timeblock.
        
        Args:
            student_id: e.g., 'S001'
            timeblock: e.g., '10:02-10:05'
            features: Feature dict with metric values
            classification: Classification result with score and components
        
        Returns:
            Narrative text from Gemini
        """
        if not self.model:
            return self._generate_default_narrative(student_id, timeblock, features, classification)
        
        # Build evidence-based prompt
        prompt = self._build_narrative_prompt(student_id, timeblock, features, classification)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_default_narrative(student_id, timeblock, features, classification)
    
    def _build_narrative_prompt(
        self,
        student_id: str,
        timeblock: str,
        features: Dict,
        classification: Dict,
    ) -> str:
        """Build factual prompt for Gemini."""
        prompt = f"""
Generate an objective, evidence-based narrative for the following classroom engagement data.
INSTRUCTION: Base your response ONLY on the provided metrics. Make NO inferences beyond the data.

STUDENT: {student_id}
TIME INTERVAL: {timeblock}

ENGAGEMENT METRICS:
- Gaze Direction: {features.get('gaze_mean_deg', 'N/A'):.1f}° (0°=facing screen, 90°=perpendicular)
- Gaze Stability: ±{features.get('gaze_std_deg', 'N/A'):.1f}°
- Posture Tilt: {features.get('posture_mean_tilt_deg', 'N/A'):.1f}° from vertical
- Posture Stability: ±{features.get('posture_stability_std_deg', 'N/A'):.1f}°
- Hand Motion: {features.get('hand_motion_mean_pxfs', 'N/A'):.1f} pixels/frame
- Gesture Frequency: {features.get('gesture_frequency_permin', 'N/A'):.1f} events/minute
- Device Interaction Index: {features.get('interaction_mean_idx', 'N/A'):.2f} (0=none, 1=active)

ENGAGEMENT CLASSIFICATION:
- Overall Score: {classification.get('engagement_score', 'N/A'):.1f}/100
- Level: {classification.get('engagement_level', 'unknown')}
- Component Scores:
  * Gaze: {classification.get('component_scores', {}).get('gaze_score', 'N/A'):.1f}/100
  * Posture: {classification.get('component_scores', {}).get('posture_score', 'N/A'):.1f}/100
  * Hand Activity: {classification.get('component_scores', {}).get('gesture_score', 'N/A'):.1f}/100
  * Device Interaction: {classification.get('component_scores', {}).get('interaction_score', 'N/A'):.1f}/100

TASK:
1. Describe what the metrics indicate (factually, no interpretation beyond data).
2. Correlate metrics with observed engagement patterns (based on thresholds only).
3. Provide teacher-actionable observations grounded in the measured data.
4. Do NOT speculate about student characteristics, abilities, or internal states.

OUTPUT: 2-3 sentences, evidence-based, neutral tone.
"""
        return prompt
    
    def _generate_default_narrative(
        self,
        student_id: str,
        timeblock: str,
        features: Dict,
        classification: Dict,
    ) -> str:
        """Fallback narrative generation (no Gemini)."""
        gaze_angle = features.get('gaze_mean_deg')
        posture_var = features.get('posture_stability_std_deg')
        gesture_freq = features.get('gesture_frequency_permin', 0.0)
        engagement_score = classification.get('engagement_score', 0)
        
        # Simple rule-based narrative
        parts = []
        parts.append(f"{student_id} ({timeblock}):")
        
        if gaze_angle is not None:
            if gaze_angle < 30:
                parts.append("maintained forward-facing gaze toward screen")
            elif gaze_angle < 60:
                parts.append("displayed partial attention (head orientation ~{:.0f}°)".format(gaze_angle))
            else:
                parts.append("gaze directed away from screen")
        
        if posture_var is not None:
            if posture_var < 5:
                parts.append("upright, stable posture")
            elif posture_var < 15:
                parts.append("moderate postural movement")
            else:
                parts.append("high postural instability")
        
        if gesture_freq > 2:
            parts.append("frequent hand gestures detected")
        elif gesture_freq > 0:
            parts.append("minimal hand activity")
        
        narrative = " ".join(parts) + f". Overall engagement: {engagement_score:.0f}/100."
        return narrative
    
    def generate_report(
        self,
        session_data: Dict,
        output_format: str = "text",
    ) -> str:
        """
        Generate complete engagement report.
        
        Args:
            session_data: Dict containing:
            {
                'session_name': str,
                'session_duration_min': float,
                'start_time': str,
                'end_time': str,
                'students_tracked': [student_ids],
                'timeline': [
                    {
                        'timeblock': 'HH:MM-HH:MM',
                        'student_id': str,
                        'features': {...},
                        'classification': {...},
                    },
                    ...
                ],
            }
            output_format: 'text', 'html', 'json'
        
        Returns:
            Formatted report string
        """
        if output_format == 'text':
            return self._generate_text_report(session_data)
        elif output_format == 'json':
            return json.dumps(session_data, indent=2, default=str)
        elif output_format == 'html':
            return self._generate_html_report(session_data)
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def _generate_text_report(self, session_data: Dict) -> str:
        """Generate text-format report."""
        lines = []
        
        lines.append("=" * 80)
        lines.append("CLASSROOM ENGAGEMENT ANALYSIS REPORT")
        lines.append("=" * 80)
        
        lines.append(f"\nSession: {session_data.get('session_name', 'Unknown')}")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration: {session_data.get('session_duration_min', 'N/A')} minutes")
        lines.append(f"Students Tracked: {len(session_data.get('students_tracked', []))}")
        
        lines.append("\n" + "-" * 80)
        lines.append("ENGAGEMENT TIMELINE")
        lines.append("-" * 80 + "\n")
        
        for record in session_data.get('timeline', []):
            timeblock = record.get('timeblock', 'N/A')
            student_id = record.get('student_id', 'N/A')
            score = record.get('classification', {}).get('engagement_score', 0)
            level = record.get('classification', {}).get('engagement_level', 'unknown')
            
            narrative = self.generate_student_narrative(
                student_id,
                timeblock,
                record.get('features', {}),
                record.get('classification', {}),
            )
            
            lines.append(f"[{timeblock}] {student_id} (Score: {score:.0f}/100, {level})")
            lines.append(f"  {narrative}\n")
        
        lines.append("-" * 80)
        lines.append("STATISTICAL SUMMARY")
        lines.append("-" * 80 + "\n")
        
        # Aggregate stats
        scores = [
            record.get('classification', {}).get('engagement_score', 0)
            for record in session_data.get('timeline', [])
        ]
        
        if scores:
            lines.append(f"Mean Engagement: {pd.Series(scores).mean():.1f}/100")
            lines.append(f"Std Deviation: {pd.Series(scores).std():.1f}")
            lines.append(f"Min Score: {pd.Series(scores).min():.1f}")
            lines.append(f"Max Score: {pd.Series(scores).max():.1f}")
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _generate_html_report(self, session_data: Dict) -> str:
        """Generate HTML-format report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Engagement Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
                h2 { color: #666; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
                th { background-color: #f0f0f0; }
                .high { background-color: #c8e6c9; }
                .moderate { background-color: #fff9c4; }
                .low { background-color: #ffccbc; }
            </style>
        </head>
        <body>
            <h1>Classroom Engagement Analysis Report</h1>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p><strong>Session:</strong> """ + session_data.get('session_name', 'Unknown') + """</p>
            <p><strong>Students Tracked:</strong> """ + str(len(session_data.get('students_tracked', []))) + """</p>
            
            <h2>Engagement Timeline</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Student</th>
                    <th>Score</th>
                    <th>Level</th>
                    <th>Summary</th>
                </tr>
        """
        
        for record in session_data.get('timeline', []):
            timeblock = record.get('timeblock', 'N/A')
            student_id = record.get('student_id', 'N/A')
            score = record.get('classification', {}).get('engagement_score', 0)
            level = record.get('classification', {}).get('engagement_level', 'unknown')
            
            # Color coding
            if level == 'engaged':
                color_class = 'high'
            elif level == 'passive':
                color_class = 'moderate'
            else:
                color_class = 'low'
            
            narrative = self.generate_student_narrative(
                student_id,
                timeblock,
                record.get('features', {}),
                record.get('classification', {}),
            )
            
            html += f"""
                <tr class="{color_class}">
                    <td>{timeblock}</td>
                    <td>{student_id}</td>
                    <td>{score:.0f}/100</td>
                    <td>{level}</td>
                    <td>{narrative}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


def create_report_generator(config: Dict) -> ReportGenerator:
    """Factory function."""
    return ReportGenerator(config)
