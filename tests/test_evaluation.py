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


class TraceEvidenceChecks(unittest.TestCase):
    def test_study_rejects_summary_not_matching_recorded_transition(self):
        # A geometrically valid summary must still match the actual hashed trace.
        import hashlib
        import json
        import tempfile
        from unittest.mock import patch
        import compare
        from insertion import CONFIG
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'configs').mkdir();(root/'artifacts/eval/run').mkdir(parents=True)
            (root/'configs/seeds.json').write_text(json.dumps({'held_out':[1000]}))
            trace=[{'state':[0.]*6,'next_state':[0.]*6,'raw_chunk':[[0.]*6],
                    'action':[0.]*6,'timestamp':0.,'max_contact_n':2.,'terminal':'COLLISION'}]
            raw=root/'1000_trace.json';raw.write_text(json.dumps(trace))
            row={k:0. for k in ('lateral_error_m','depth_error_m','orientation_error_rad','duration_s',
                                  'path_length_m','inference_ms','clipped_steps')}
            row.update(seed=1000,outcome='COLLISION',steps=1,max_contact_n=2.,
                       initial_state=[0.]*6,final_state=[0.]*6,raw_artifact_dir='.',
                       trace_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            summary={'environment_config':CONFIG,'execution_status':'PASS','split':'held_out',
                     'policy':'act','action_chunk_size':1,**aggregate([row])}
            folder=root/'artifacts/eval/run'
            (folder/'summary.json').write_text(json.dumps(summary))
            (folder/'episodes.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(compare,'ROOT',root):
                compare.verified_run('run',require_transitions=True)
                row['final_state'][0]=.02
                (folder/'episodes.jsonl').write_text(json.dumps(row)+'\n')
                with self.assertRaises(AssertionError):
                    compare.verified_run('run',require_transitions=True)
                row['final_state'][0]=0.
                (folder/'episodes.jsonl').write_text(json.dumps(row)+'\n')
                del trace[0]['next_state'];raw.write_text(json.dumps(trace))
                row['trace_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
                (folder/'episodes.jsonl').write_text(json.dumps(row)+'\n')
                with self.assertRaisesRegex(ValueError,'DETAILED_TRANSITIONS_REQUIRED'):
                    compare.verified_run('run',require_transitions=True)
