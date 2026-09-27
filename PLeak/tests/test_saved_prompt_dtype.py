import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class SavedPromptDtypeTests(unittest.TestCase):
    def test_both_generation_paths_cast_only_soft_prompts(self):
        tree = ast.parse((ROOT / 'prompt_obfuscation/src/output_generation.py').read_text())
        for name in ('generate_model_responses_replace', 'precompute_model_outputs_replace'):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            loop = next(n for n in fn.body if isinstance(n, ast.For))
            start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'prompt_for_generation')
            statements = loop.body[start:start + 3]
            code = compile(ast.Module(body=statements, type_ignores=[]), '<replacement>', 'exec')
            for soft in (False, True):
                for dtype in ('float16', 'bfloat16', 'float32'):
                    with self.subTest(path=name, soft=soft, dtype=dtype):
                        ids = object()
                        embeddings = SimpleNamespace(dtype=dtype, device='cpu')
                        converted = object()
                        saved = SimpleNamespace(dtype='float32', to=Mock(return_value=converted))
                        get_embeddings = Mock(return_value=embeddings)
                        replace = Mock(return_value='result')
                        ns = dict(sys_prompt_obf=saved, is_soft_prompt_method=soft,
                                  input_batch=ids, sys_prompt_indices_batch=[(1, 3)],
                                  model_wrapper=SimpleNamespace(get_embeddings=get_embeddings),
                                  replace_sys_prompt_batch=replace)
                        exec(code, ns)
                        if soft:
                            saved.to.assert_called_once_with(device='cpu', dtype=dtype)
                            replace.assert_called_once_with(converted, embeddings, [(1, 3)])
                        else:
                            saved.to.assert_not_called()
                            get_embeddings.assert_not_called()
                            replace.assert_called_once_with(saved, ids, [(1, 3)])
                        self.assertIs(ns['sys_prompt_obf'], saved)
                        self.assertEqual(saved.dtype, 'float32')


if __name__ == '__main__':
    unittest.main()
