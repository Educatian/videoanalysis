"""Synthetic report fixtures only; no videos, models or remote services."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'report_generator', Path(__file__).parents[1] / 'src' / 'report_generator.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportHtmlTests(unittest.TestCase):
    def setUp(self):
        self.generator = module.ReportGenerator({'report_generation': {'enabled': False}})
        self.fixture = {
            'session_name': 'Practice <group> & discussion',
            'students_tracked': ['P01'],
            'timeline': [{
                'timeblock': '0 < t < 10', 'student_id': 'P01 & partner',
                'features': {},
                'classification': {'engagement_score': 75, 'engagement_level': 'engaged'},
            }],
        }

    def test_session_and_row_labels_remain_literal_text(self):
        report = self.generator.generate_report(self.fixture, 'html')
        self.assertIn('Practice &lt;group&gt; &amp; discussion', report)
        self.assertIn('<td>0 &lt; t &lt; 10</td>', report)
        self.assertIn('<td>P01 &amp; partner</td>', report)

    def test_generated_narrative_is_text_within_its_table_cell(self):
        self.generator.generate_student_narrative = lambda *args: 'Score < threshold & stable'
        report = self.generator.generate_report(self.fixture, 'html')
        self.assertIn('<td>Score &lt; threshold &amp; stable</td>', report)
        self.assertIn('<tr class="high">', report)
        self.assertIn('<td>75/100</td>', report)

    def test_plain_text_report_stays_plain_text(self):
        report = self.generator.generate_report(self.fixture, 'text')
        self.assertIn('Practice <group> & discussion', report)


if __name__ == '__main__':
    unittest.main()
