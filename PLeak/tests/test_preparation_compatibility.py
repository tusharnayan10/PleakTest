import ast
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'PLeak'))
from prepare_defenses import run_artifact_stage


class CompatibilityTests(unittest.TestCase):
    def resize(self, backend, size):
        tree = ast.parse((ROOT / 'prompt_obfuscation/src/model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_resize_token_embeddings')
        ns = {'inspect': inspect}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<resize>', 'exec'), ns)
        return ns[method.name](SimpleNamespace(model=backend), size)

    def test_old_transformers_signature(self):
        class Old:
            def resize_token_embeddings(self, size):
                return size
        self.assertEqual(self.resize(Old(), 32001), 32001)

    def test_new_transformers_disables_mean_initialization(self):
        class New:
            def resize_token_embeddings(self, size, mean_resizing=True):
                return size, mean_resizing
        self.assertEqual(self.resize(New(), 32001), (32001, False))

    def test_internal_type_error_is_not_swallowed(self):
        class Broken:
            def resize_token_embeddings(self, size, mean_resizing=True):
                raise TypeError('internal failure')
        with self.assertRaisesRegex(TypeError, 'internal failure'):
            self.resize(Broken(), 32001)

    def test_success_exit_without_artifact_is_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder, patch('prepare_defenses.subprocess.run'):
            with self.assertRaisesRegex(RuntimeError, 'first ERROR/traceback'):
                run_artifact_stage(['python', 'obfuscate.py'], [Path(folder) / 'params.json'])

    def test_success_requires_all_artifacts(self):
        with tempfile.TemporaryDirectory() as folder, patch('prepare_defenses.subprocess.run') as run:
            output = Path(folder) / 'best_candidate.pt'
            output.write_bytes(b'test')
            run_artifact_stage(['python', 'evaluate_obfuscation.py'], [output])
            self.assertTrue(run.call_args.kwargs['check'])

    def test_nonzero_exit_propagates(self):
        with patch('prepare_defenses.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'python')):
            with self.assertRaises(subprocess.CalledProcessError):
                run_artifact_stage(['python', 'obfuscate.py'], [])


if __name__ == '__main__':
    unittest.main()
