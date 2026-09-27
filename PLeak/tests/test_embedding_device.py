import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class EmbeddingDeviceTests(unittest.TestCase):
    def test_ids_follow_weight_device_and_result_returns_to_cpu(self):
        tree = ast.parse((ROOT / 'prompt_obfuscation/src/model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_embeddings')
        ns = {'torch': SimpleNamespace(Tensor=object)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<embedding>', 'exec'), ns)
        for device in ('cpu', 'cuda:0', 'cuda:3'):
            with self.subTest(device=device):
                moved_ids = object()
                ids = SimpleNamespace(to=Mock(return_value=moved_ids))
                result = SimpleNamespace(dtype='float16', device='cpu')
                selected = SimpleNamespace(cpu=Mock(return_value=result))
                class Weight:
                    def __getitem__(self, index):
                        if index is not moved_ids:
                            raise AssertionError('Lookup did not use device-aligned indices')
                        return selected
                weight = Weight()
                weight.device = device
                wrapper = SimpleNamespace(model=SimpleNamespace(
                    get_input_embeddings=lambda: SimpleNamespace(weight=weight)))
                self.assertIs(ns['get_embeddings'](wrapper, ids), result)
                ids.to.assert_called_once_with(device=device)
                selected.cpu.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
