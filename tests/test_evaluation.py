"""Evaluation arithmetic must retain failed episodes in the denominator."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/local'))
from evaluate import aggregate


class EvaluationChecks(unittest.TestCase):
    def test_failure_counts_and_confidence_interval(self):
        base={k:1. for k in ('lateral_error_m','depth_error_m','orientation_error_rad','duration_s',
                             'path_length_m','inference_ms','clipped_steps')}
        result=aggregate([{**base,'outcome':'SUCCESS'},{**base,'outcome':'COLLISION'},
                          {**base,'outcome':'TIMEOUT'},{**base,'outcome':'TIMEOUT'}])
        self.assertEqual(result['episodes'],4)
        self.assertEqual(result['success_rate'],.25)
        self.assertEqual(result['collision_rate'],.25)
        self.assertEqual(result['timeout_rate'],.5)
        lo,hi=result['success_wilson_95']
        self.assertAlmostEqual(lo,.04558726,places=6)
        self.assertAlmostEqual(hi,.69935816,places=6)
