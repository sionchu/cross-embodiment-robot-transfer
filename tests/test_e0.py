import importlib.util
from pathlib import Path
import unittest
import json
import subprocess
import tempfile
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'e0', Path(__file__).resolve().parents[1] / 'scripts/colab/e0.py')
e0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e0)


class EvidenceChecks(unittest.TestCase):
    good = ('INFO:root:MSE for trajectory 1: 0.1, MAE: 0.2\n'
            'INFO:root:MSE for trajectory 2: 1e-3, MAE: 0.04\n')

    def test_preflight_failure_preserves_failed_manifest_without_command_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            failed = subprocess.CompletedProcess(['nvidia-smi'], 1, 'sensitive-output')
            with patch.object(e0.subprocess, 'run', return_value=failed):
                with self.assertRaises(RuntimeError):
                    e0.run(Path(tmp), output)
            report = json.loads((output / 'e0.json').read_text())
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['evaluation_episode_count'], 0)
            self.assertEqual(report['runs'], [])
            self.assertNotIn('sensitive-output', (output / 'e0.json').read_text())
            with self.assertRaises(FileExistsError):
                e0.run(Path(tmp), output)

    def test_two_finite_episode_results(self):
        result = e0.parse_metrics(self.good, 0)
        self.assertEqual([r['episode_id'] for r in result], [1, 2])
        self.assertEqual(result[1]['mse'], 0.001)

    def test_process_failure_cannot_pass(self):
        with self.assertRaises(ValueError):
            e0.parse_metrics(self.good, 1)

    def test_missing_duplicate_or_wrong_episodes_cannot_pass(self):
        for log in ('', self.good.splitlines()[0], self.good + self.good,
                    self.good.replace('trajectory 2', 'trajectory 0')):
            with self.subTest(log=log), self.assertRaises(ValueError):
                e0.parse_metrics(log, 0)

    def test_nonfinite_negative_or_invalid_metrics_cannot_pass(self):
        for value in ('nan', 'inf', '-inf', '-0.1', 'broken'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                e0.parse_metrics(self.good.replace('0.1,', value + ','), 0)


if __name__ == '__main__':
    unittest.main()
