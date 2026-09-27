import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


class GenerationDeviceTests(unittest.TestCase):
    def test_generation_inputs_follow_model_device_with_beams_preserved(self):
        tree = ast.parse((ROOT / 'prompt_obfuscation/src/model.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_output')
        ns = {'torch': SimpleNamespace(Tensor=object, no_grad=nullcontext)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<generate>', 'exec'), ns)
        for embedded in (False, True):
            with self.subTest(embedded=embedded):
                moved = SimpleNamespace(device='cuda:7')
                moved_mask = SimpleNamespace(device='cuda:7')
                inputs = SimpleNamespace(to=Mock(return_value=moved), device='cuda:0')
                mask = SimpleNamespace(to=Mock(return_value=moved_mask), device='cuda:0')
                final = object()
                sequences = SimpleNamespace(shape=(1, 4), view=Mock(return_value=final))
                output = SimpleNamespace(cpu=lambda: sequences)
                def generate(**kwargs):
                    self.assertIs(kwargs['inputs_embeds' if embedded else 'input_ids'], moved)
                    self.assertIs(kwargs['attention_mask'], moved_mask)
                    self.assertEqual(kwargs['num_beams'], 3)
                    self.assertEqual(kwargs['temperature'], 0.9)
                    self.assertEqual(kwargs['top_p'], 0.6)
                    self.assertEqual(kwargs['max_new_tokens'], 80)
                    return output
                backend = SimpleNamespace(model=SimpleNamespace(device='cuda:7', generate=generate),
                    tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2))
                settings = dict(num_beams=3, do_sample=True, temperature=0.9, top_p=0.6, max_new_tokens=80)
                self.assertIs(ns['generate_output'](backend, inputs, mask, settings, embedded), final)
                inputs.to.assert_called_once_with(device='cuda:7')
                mask.to.assert_called_once_with(device='cuda:7')


if __name__ == '__main__':
    unittest.main()
