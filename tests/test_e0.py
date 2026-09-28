import importlib.util
from pathlib import Path
import unittest
import json
import subprocess
import tempfile
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    'e0', Path(__file__).resolve().parents[1] / 'scripts/colab/e0.py')
e0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e0)


class EvidenceChecks(unittest.TestCase):
    good = ('INFO:root:MSE for trajectory 1: 0.1, MAE: 0.2\n'
            'INFO:root:MSE for trajectory 2: 1e-3, MAE: 0.04\n')

    def test_log_redaction_preserves_failure_diagnostics(self):
        raw = ('Traceback (most recent call last):\n'
               'huggingface_hub.errors.OfflineModeIsEnabled: model_info failed\n'
               'HF_TOKEN=opaque-test-secret legacy=legacy-test-secret hf_short\n'
               'Authorization: Bearer opaque-bearer-value\n'
               'Authorization: Basic dXNlcjpwYXNz\n'
               'https://cdn.example/model?X-Amz-Credential=private&Signature=private\n'
               'RuntimeError: CUDA out of memory\n')
        safe = e0.redact_log(raw, ('opaque-test-secret', 'legacy-test-secret', ''))
        for secret in ('opaque-test-secret', 'legacy-test-secret', 'hf_short',
                       'opaque-bearer-value', 'dXNlcjpwYXNz', 'private'):
            self.assertNotIn(secret, safe)
        for diagnostic in ('Traceback', 'OfflineModeIsEnabled: model_info failed',
                           'RuntimeError: CUDA out of memory', 'https://cdn.example/model'):
            self.assertIn(diagnostic, safe)

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

    def test_known_preflight_failure_has_fixed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            ok = subprocess.CompletedProcess(['nvidia-smi'], 0, 'mock hardware')
            with patch.object(e0.subprocess, 'run', return_value=ok), \
                    patch.object(e0.platform, 'system', return_value='Windows'):
                with self.assertRaises(e0.E0Failure):
                    e0.run(Path(tmp), output)
            report = json.loads((output / 'e0.json').read_text())
            self.assertEqual(report['status'], 'FAILED')
            self.assertEqual(report['stage'], 'preflight')
            self.assertEqual(report['failure_code'], 'WRONG_PLATFORM')

    def test_unknown_and_sensitive_exceptions_are_sanitized(self):
        sensitive = 'hf_FAKE_TEST_SECRET /private/local/path Traceback: raw failure'
        for exc, expected in (
                (RuntimeError(sensitive), 'UNKNOWN_FAILURE'),
                (e0.E0Failure('GPU_MEMORY_INSUFFICIENT', sensitive), 'GPU_MEMORY_INSUFFICIENT'),
                (e0.E0Failure(sensitive, sensitive), 'UNKNOWN_FAILURE')):
            with self.subTest(error_type=type(exc).__name__, code=expected), \
                    tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'run'
                with patch.object(e0.subprocess, 'run', side_effect=exc):
                    with self.assertRaises(RuntimeError):
                        e0.run(Path(tmp), output)
                text = (output / 'e0.json').read_text()
                report = json.loads(text)
                self.assertEqual(report['status'], 'FAILED')
                self.assertEqual(report['failure_code'], expected)
                self.assertEqual(report['error_type'], type(exc).__name__)
                for fragment in (sensitive, 'hf_FAKE_TEST_SECRET', '/private/local/path',
                                 'Traceback', 'raw failure'):
                    self.assertNotIn(fragment, text)

    def test_mocked_runtime_classification_and_success(self):
        # Exercise the wrapper only: no real GPU, package import or download.
        pins = json.loads((e0.ROOT / 'references/source_manifest.json').read_text())

        class GatedRepoError(RuntimeError):
            pass

        for case, expected in (
                ('success', None), ('t4', None), ('gated', 'HF_GATED_ACCESS'),
                ('cuda', 'CUDA_UNAVAILABLE'), ('vram', 'GPU_MEMORY_INSUFFICIENT'),
                ('smoke', 'CUDA_SMOKE_FAILED'),
                ('exit', 'OFFICIAL_INFERENCE_FAILED'),
                ('second_exit', 'OFFICIAL_INFERENCE_FAILED'),
                ('metric', 'INFERENCE_METRIC_INVALID')):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp, \
                    ExitStack() as stack:
                upstream = Path(tmp) / 'upstream'
                sample = upstream / 'demo_data/droid_sample'
                sample.mkdir(parents=True)
                (sample / 'episode.parquet').write_bytes(b'mocked sample')
                (sample / 'episode.mp4').write_bytes(b'mocked video')
                (upstream / 'uv.lock').write_bytes(b'mocked lock')
                torch = Mock(__version__=pins['environment']['torch'])
                torch.version = SimpleNamespace(cuda='mocked')
                torch.cuda.is_available.return_value = case != 'cuda'
                torch.cuda.get_device_name.return_value = 'mock NVIDIA CUDA GPU'
                torch.cuda.get_device_properties.return_value = SimpleNamespace(
                    total_memory=(12_000_000_000 if case == 'vram'
                                  else 15_637_086_208 if case == 't4'
                                  else 24_000_000_000)
                )
                torch.ones.return_value.sum.return_value.item.return_value = 1
                if case == 'smoke':
                    torch.ones.side_effect = RuntimeError('sensitive CUDA detail')
                hub = Mock()
                hub.HfApi.return_value.model_info.return_value.sha = pins['backbone']['revision']
                hub.snapshot_download.side_effect = lambda repo, revision: str(
                    Path(tmp) / (pins['backbone']['revision'] if revision == 'main' else revision))
                if case == 'gated':
                    hub.hf_hub_download.side_effect = GatedRepoError('hf_FAKE_TEST_SECRET /private/path')

                def command(args, **kwargs):
                    stdout, code = 'mock command output', 0
                    if args[:2] == ['git', 'rev-parse']:
                        stdout = pins['upstream']['commit']
                    elif args[:2] in (['git', 'status'], ['git', 'submodule']):
                        stdout = ''
                    elif args[0] == 'ffmpeg':
                        stdout = 'ffmpeg version 7.0'
                    elif args[0] == e0.sys.executable:
                        self.assertNotIn('HF_TOKEN', kwargs['env'])
                        self.assertNotIn('HUGGING_FACE_HUB_TOKEN', kwargs['env'])
                        self.assertEqual(kwargs['env'].get('HF_HUB_OFFLINE'), '1')
                        self.assertEqual(kwargs['env'].get('TRANSFORMERS_OFFLINE'), '1')
                        self.assertEqual(kwargs['env'].get('GROOT_HF_LOCAL_FIRST'), '1')
                        self.assertEqual(kwargs['env'].get('GROOT_PATCH_MISTRAL'), '1')
                        self.assertEqual(kwargs['env'].get('GROOT_SKIP_HF_MODEL_WEIGHTS'), '0')
                        self.assertEqual(kwargs['env'].get('HF_HUB_DISABLE_IMPLICIT_TOKEN'), '1')
                        stdout = (self.good + '\nsecret=hf_FAKE_TEST_SECRET '
                                  'other=hf_OTHER_FAKE_TOKEN_123456 '
                                  'opaque-test-secret legacy-test-secret'
                                  if case != 'metric'
                                  else 'invalid metrics: hf_FAKE_TEST_SECRET')
                        code = int(case == 'exit' or (
                            case == 'second_exit' and pins['models'][1]['revision'] in args[3]))
                        if code:
                            stdout += ('\nTraceback (most recent call last):\n'
                                       'OfflineModeIsEnabled: model_info failed\n')
                    return subprocess.CompletedProcess(args, code, stdout)

                real_digest = e0.digest
                stack.enter_context(patch.object(e0, 'digest', side_effect=lambda path:
                    pins['upstream']['lock_sha256'] if Path(path).name == 'uv.lock'
                    else real_digest(path)))
                stack.enter_context(patch.object(e0.subprocess, 'run', side_effect=command))
                stack.enter_context(patch.object(e0.platform, 'system', return_value='Linux'))
                stack.enter_context(patch.object(e0.platform, 'machine', return_value='x86_64'))
                stack.enter_context(patch.object(e0.sys, 'version_info', (3, 12)))
                stack.enter_context(patch.object(e0.importlib.metadata, 'version',
                                                return_value=pins['environment']['torchcodec']))
                stack.enter_context(patch.object(e0.importlib.metadata, 'distributions', return_value=[]))
                stack.enter_context(patch.dict(e0.sys.modules, {
                    'torch': torch, 'torchcodec': Mock(), 'huggingface_hub': hub,
                    'huggingface_hub.errors': SimpleNamespace(GatedRepoError=GatedRepoError),
                }))
                stack.enter_context(patch.dict(e0.os.environ, {
                    'HF_TOKEN': 'opaque-test-secret',
                    'HUGGING_FACE_HUB_TOKEN': 'legacy-test-secret',
                    'GROOT_SKIP_HF_MODEL_WEIGHTS': '1',
                }))
                output = Path(tmp) / 'run'
                if expected:
                    with self.assertRaises(RuntimeError):
                        e0.run(upstream, output)
                else:
                    e0.run(upstream, output)
                text = (output / 'e0.json').read_text()
                report = json.loads(text)
                self.assertNotIn('hf_FAKE_TEST_SECRET', text)
                self.assertNotIn('/private/path', text)
                self.assertNotIn('sensitive CUDA detail', text)
                for log_path in output.glob('*.log'):
                    log_text = log_path.read_text()
                    self.assertNotIn('hf_FAKE_TEST_SECRET', log_text)
                    self.assertNotIn('hf_OTHER_FAKE_TOKEN_123456', log_text)
                    self.assertNotIn('opaque-test-secret', log_text)
                    self.assertNotIn('legacy-test-secret', log_text)
                for run in report['runs']:
                    log_path = output / run['log_file']
                    self.assertEqual(run['log_sha256'], real_digest(log_path))
                if expected:
                    self.assertEqual(report['status'], 'FAILED')
                    self.assertEqual(report['failure_code'], expected)
                    if case == 'gated':
                        self.assertEqual(report['error_type'], 'GatedRepoError')
                        self.assertEqual(report['stage'], 'gated_access_and_snapshots')
                    if case in ('exit', 'second_exit'):
                        failed_run = report['runs'][-1]
                        self.assertEqual(failed_run['status'], 'FAILED')
                        self.assertEqual(failed_run['exit_code'], 1)
                        self.assertNotIn('episodes', failed_run)
                        self.assertIn('OfflineModeIsEnabled: model_info failed',
                                      (output / failed_run['log_file']).read_text())
                        self.assertEqual(report['evaluation_episode_count'],
                                         2 if case == 'second_exit' else 0)
                else:
                    self.assertEqual(report['status'], 'PASS')
                    self.assertEqual(report['evaluation_episode_count'], 4)
                    self.assertNotIn('failure_code', report)
                    for run in report['runs']:
                        self.assertEqual(run['episodes'], e0.parse_metrics(self.good, 0))
                        self.assertNotIn('failure_code', run)

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
