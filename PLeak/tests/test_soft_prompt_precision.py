"""Run the official soft optimizer initialization and replacement with tiny tensors."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]


def statements():
    tree = ast.parse((ROOT / 'prompt_obfuscation/obfuscate.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'obfuscate_soft_prompt')
    init = fn.body[1:4]  # after docstring: master parameter, requires_grad, Adam
    replace = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == 'replace_sys_prompt_batch')
    return init, replace


class PrecisionStructureTests(unittest.TestCase):
    def test_master_is_float32_before_optimizer(self):
        init, replacement = statements()
        self.assertEqual(init[0].value.func.attr, 'float')
        self.assertEqual(init[1].value.func.attr, 'requires_grad_')
        self.assertEqual(init[2].value.func.attr, 'Adam')
        self.assertEqual(replacement.value.args[0].func.attr, 'to')
        self.assertEqual(replacement.value.args[0].keywords[0].arg, 'dtype')


@unittest.skipIf(torch is None, 'PyTorch is required for the numerical optimizer test')
class PrecisionNumericalTests(unittest.TestCase):
    def test_cpu_half_forward_updates_fp32_master(self):
        for dtype in (torch.float16, torch.float32):
            initial = torch.tensor([[0.25, -0.5]], dtype=dtype)
            ns = {'torch': torch, 'model_wrapper': SimpleNamespace(get_embeddings=lambda ids: initial),
                  'sys_prompt_obf': None, 'lr': 0.01}
            init, replacement = statements()
            exec(compile(ast.Module(body=init, type_ignores=[]), '<official-init>', 'exec'), ns)
            master = ns['sys_prompt_obf_emb']
            self.assertEqual(master.dtype, torch.float32)
            self.assertTrue(master.is_leaf)
            ns.update(base_embedded_input_ids=torch.zeros((1, 3, 2), dtype=dtype),
                      sys_prompt_indices_batch=[(1, 2)])
            # Execute official replacement helpers too, without importing model deps.
            tree = ast.parse((ROOT / 'prompt_obfuscation/src/prompt_utils.py').read_text())
            helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name in ('replace_sys_prompt', 'replace_sys_prompt_batch')]
            exec(compile(ast.Module(body=helpers, type_ignores=[]), '<official-replacement>', 'exec'), ns)
            exec(compile(ast.Module(body=[replacement], type_ignores=[]), '<official-cast>', 'exec'), ns)
            forward = ns['base_embedded_input_ids']
            self.assertEqual(forward.dtype, dtype)
            before = master.detach().clone()
            forward.float().square().sum().backward()
            self.assertEqual(master.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(master.grad).all())
            ns['optimizer'].step()
            self.assertFalse(torch.equal(before, master))
            self.assertTrue(torch.isfinite(master).all())
            self.assertEqual(ns['optimizer'].state[master]['exp_avg'].dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
