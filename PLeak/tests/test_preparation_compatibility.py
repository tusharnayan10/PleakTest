import ast
from contextlib import nullcontext
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
from prepare_defenses import run_artifact_stage, validate_obfuscation_context


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


class CacheCompatibilityTests(unittest.TestCase):
    def generate(self, modern, embedded):
        tree = ast.parse((ROOT / 'prompt_obfuscation/src/model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_logits')
        ns = {'torch': SimpleNamespace(Tensor=object, no_grad=nullcontext)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<generation>', 'exec'), ns)
        config = SimpleNamespace(return_legacy_cache=True) if modern else SimpleNamespace()
        backend = SimpleNamespace(generation_config=config, generate=lambda **kw: kw)
        wrapper = SimpleNamespace(model=backend, tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2))
        result = ns['generate_logits'](wrapper, 'input', 'mask', 15, embedded)
        self.assertEqual(result['max_new_tokens'], 15)
        self.assertEqual(result['num_beams'], 1)
        self.assertFalse(result['do_sample'])
        self.assertTrue(result['output_scores'])
        self.assertEqual(result['inputs_embeds' if embedded else 'input_ids'], 'input')
        return result

    def test_legacy_cache_option_is_omitted(self):
        for embedded in (False, True):
            self.assertNotIn('return_legacy_cache', self.generate(False, embedded))

    def test_modern_cache_option_is_preserved(self):
        for embedded in (False, True):
            self.assertIs(self.generate(True, embedded)['return_legacy_cache'], False)

    def test_exact_context_is_accepted(self):
        validate_obfuscation_context({'system_prompt': '<|pad|>original\n<|pad|>'}, 'original\n')

    def test_added_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fresh --output-dir'):
            validate_obfuscation_context({'system_prompt': '<|pad|>QA instruction original<|pad|>'}, 'original')

    def test_preparation_passes_official_task_hints_flag(self):
        source = (ROOT / 'PLeak/prepare_defenses.py').read_text()
        tree = ast.parse(source)
        commands = [n for n in ast.walk(tree) if isinstance(n, ast.List)
                    and any(isinstance(x, ast.Constant) and x.value == 'obfuscate.py' for x in n.elts)]
        self.assertEqual(len(commands), 1)
        self.assertIn('--task_hints', [x.value for x in commands[0].elts if isinstance(x, ast.Constant)])


if __name__ == '__main__':
    unittest.main()
