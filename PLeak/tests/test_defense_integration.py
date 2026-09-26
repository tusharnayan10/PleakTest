"""Dependency-free routing/regression tests; real inference: smoke_defenses.py."""
import ast
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

PLEAK = Path(__file__).resolve().parents[1]
ROOT = PLEAK.parent
sys.path.insert(0, str(PLEAK))
from DefenseAdapters import Artifacts, PromptKeeperWrapper, PromptObfuscationWrapper, context_key, make_wrapper
from DefenseCLI import parse_args, result_path

BASE = '822a3edce94b34a82d965e0c67d40e81b2601ec1'


def function(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream>', 'exec'), namespace)
    return namespace[name]


class RegressionTests(unittest.TestCase):
    def test_attack_file_unchanged(self):
        old = subprocess.check_output(['git', 'show', f'{BASE}:PLeak/Attack.py'], cwd=ROOT)
        self.assertEqual(old, (PLEAK / 'Attack.py').read_bytes())

    def test_metrics_postprocessing_and_csv_unchanged(self):
        old = subprocess.check_output(['git', 'show', f'{BASE}:PLeak/Sampler.py'], cwd=ROOT).decode()
        new = (PLEAK / 'Sampler.py').read_text()
        def methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
            return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        before, after = methods(old), methods(new)
        for name in ('evaluate', 'postprocess', 'filter_tokens', 'sentence_to_char', 'sentence_to_tokens', 'save_to_csv'):
            self.assertEqual(before[name], after[name], name)

    def test_cli_defaults_and_legacy_positionals(self):
        with patch.object(sys, 'argv', ['sample.py', 'Roles', 'llama-chat', 'None', 'AQ']):
            args = parse_args()
        self.assertEqual((args.dataset, args.model, args.defense, args.triggers), ('Roles', 'llama-chat', 'None', 'AQ'))
        with patch.object(sys, 'argv', ['main.py', 'Roles', '12', 'llama', 'llama-chat', '16']):
            self.assertEqual(parse_args(training=True).defense, 'None')

    def test_artifact_model_context_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'model': 'victim', 'contexts': {context_key('secret'): {'tensor': 'prompt.pt'}}}))
            artifacts = Artifacts(path, 'victim')
            self.assertEqual(artifacts.resolve(artifacts.entry('secret')['tensor']), (Path(directory) / 'prompt.pt').resolve())
            with self.assertRaises(ValueError):
                artifacts.entry('other')
            with self.assertRaises(ValueError):
                Artifacts(path, 'different model')

    def test_no_invented_sandwich_or_fallback(self):
        self.assertIsNone(make_wrapper(None, 'None', None))
        self.assertIsNone(make_wrapper(None, 'Filter', None))
        with self.assertRaises(ValueError):
            make_wrapper(None, 'Sandwich', None)
        with self.assertRaises(ValueError):
            make_wrapper(None, 'PromptKeeper', None)

    def test_csv_paths_cannot_be_derived_from_query_slashes(self):
        with tempfile.TemporaryDirectory() as directory, patch('DefenseCLI.Path.mkdir'):
            args = SimpleNamespace(output=None, dataset='Roles', model='llama', defense='PromptKeeper')
            path = result_path(args, '../arbitrary/query')
            self.assertEqual(Path(path).parent, Path('results/PromptKeeper'))


class DefaultMunch(dict):
    __getattr__ = dict.get
    @classmethod
    def fromDict(cls, value):
        return cls(value)


class KeeperRoutingTests(unittest.TestCase):
    def run_case(self, ratio):
        # Execute upstream's actual complete decision/mitigation function, with
        # deterministic model and likelihood boundaries; do not duplicate it.
        calls = []
        def generation(**kwargs):
            calls.append(kwargs)
            return 'candidate secret' if kwargs.get('system_prompt') else 'regenerated safe answer'
        module = ModuleType('official_defense_test')
        module.__dict__.update({
            'get_text_generation': generation,
            'get_hypothesis_threshold_for_mll_test_regen': lambda **kw: (1., 'norm', [0., 1.], [1., 2.]),
            'get_likelihood_ratio': lambda **kw: (ratio, 'likelihood info'),
        })
        actual = function((ROOT / 'PromptKeeper/promptkeeper/defense.py').read_text(),
                          'get_defended_model_response', module.__dict__)
        wrapper = PromptKeeperWrapper.__new__(PromptKeeperWrapper)
        wrapper.sampler = SimpleNamespace(model_id='same-victim')
        wrapper.artifacts = SimpleNamespace(data={'significance': 0.05})
        wrapper.threshold = Mock(return_value=(1., 'norm', [0., 1.], [1., 2.]))
        wrapper.backend = lambda: nullcontext()
        wrapper.defense = module
        with patch.dict(sys.modules, {'munch': SimpleNamespace(DefaultMunch=DefaultMunch)}):
            response = wrapper.generate('secret', 'same AQ', {
                'input_ids': SimpleNamespace(shape=(1, 8)), 'max_length': 66,
                'num_beams': 3, 'temperature': 0.9, 'top_p': 0.6})
        wrapper.threshold.assert_called_once()
        self.assertEqual(calls[0]['model_id'], 'same-victim')
        self.assertEqual(calls[0]['user_query'], 'same AQ')
        self.assertEqual(calls[0]['generate_kwargs']['max_new_tokens'], 58)
        self.assertEqual(calls[0]['generate_kwargs']['num_beams'], 3)
        return response, calls

    def test_safe_candidate_is_returned(self):
        response, calls = self.run_case(0.1)
        self.assertEqual(response, 'candidate secret')
        self.assertEqual(len(calls), 1)

    def test_detected_leak_returns_actual_regeneration(self):
        response, calls = self.run_case(2.)
        self.assertEqual(response, 'regenerated safe answer')
        self.assertEqual(len(calls), 2)
        self.assertNotIn('system_prompt', calls[1])
        self.assertEqual(calls[0]['generate_kwargs'], calls[1]['generate_kwargs'])

    def test_threshold_equality_uses_upstream_regeneration(self):
        response, calls = self.run_case(1.)
        self.assertEqual(response, 'regenerated safe answer')


class Tensor:
    """Minimal token container for Sampler routing without importing torch."""
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))
    def __getitem__(self, key):
        return self.values[key[1]]


class SamplerRoutingTests(unittest.TestCase):
    def run_case(self, defense):
        tree = ast.parse((PLEAK / 'Sampler.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_sequence')
        ns = {'torch': SimpleNamespace(no_grad=nullcontext)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<sampler>', 'exec'), ns)
        tokens = Tensor([10, 11, 12])
        tokenizer = Mock(return_value=SimpleNamespace(to=lambda device: SimpleNamespace(input_ids=tokens)))
        tokenizer.eos_token_id = 2
        tokenizer.decode.return_value = 'raw baseline'
        wrapper = None if defense == 'None' else SimpleNamespace(generate=Mock(return_value='actual defended response'))
        sampler = SimpleNamespace(tokenizer=tokenizer, target_model='llama-chat', device='cpu',
            template=SimpleNamespace(format_trigger=lambda text: 'prefix:' + text + '\n'),
            model=SimpleNamespace(generate=Mock(return_value=Tensor([10, 11, 12, 99]))),
            postprocess=Mock(side_effect=lambda response, query: response + ' processed'),
            defender=SimpleNamespace(defend=lambda name, target, output: output),
            defense=defense, victim=wrapper, evaluate=Mock())
        with patch('builtins.print'):
            result = ns['sample_sequence'](sampler, ['original context'], 'AQ')
        self.assertEqual(result[0]['context'], 'original context')
        if wrapper:
            wrapper.generate.assert_called_once()
            args = wrapper.generate.call_args.args
            self.assertEqual(args[:2], ('original context', 'prefix:AQ\n'))
            self.assertEqual(args[2]['max_length'], 56)
            self.assertEqual(result[0]['AQ'], 'actual defended response processed')
            sampler.model.generate.assert_not_called()
        else:
            sampler.model.generate.assert_called_once()
            self.assertEqual(result[0]['AQ'], 'raw baseline processed')
        return sampler

    def test_none_uses_original_model_call(self):
        self.run_case('None')

    def test_obfuscation_response_reaches_existing_pipeline(self):
        self.run_case('PromptObfuscation')

    def test_keeper_response_reaches_existing_pipeline(self):
        self.run_case('PromptKeeper')



class MiniTensor:
    def __init__(self, values):
        self.values = values
        self.shape = self._shape(values)
        self.ndim = len(self.shape)
        self.dtype = 'test-dtype'
    @staticmethod
    def _shape(value):
        return (len(value),) + MiniTensor._shape(value[0]) if isinstance(value, list) and value else ()
    def __len__(self):
        return len(self.values)
    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        def get(value, keys):
            if not keys:
                return value
            part = value[keys[0]]
            if isinstance(keys[0], slice):
                return [get(v, keys[1:]) for v in part]
            return get(part, keys[1:])
        return MiniTensor(get(self.values, indices))
    def __iter__(self):
        return (MiniTensor(value) for value in self.values)
    def cpu(self):
        return self
    def to(self, *args, **kwargs):
        return self
    def item(self):
        return self.values


class ObfuscationRoutingTests(unittest.TestCase):
    def run_case(self, soft):
        fake_torch = SimpleNamespace(
            equal=lambda a, b: a.values == b.values,
            cat=lambda tensors, dim=0: MiniTensor([v for t in tensors for v in t.values]),
            stack=lambda tensors, dim=0: MiniTensor([t.values for t in tensors]),
            ones=lambda count: MiniTensor([1] * count),
            ones_like=lambda ids: MiniTensor([[1] * ids.shape[1]]))
        source = (ROOT / 'prompt_obfuscation/src/prompt_utils.py').read_text()
        ns = {'torch': fake_torch}
        # Postpone annotations so tests do not need torch's runtime types.
        ns['__builtins__'] = __builtins__
        tree = ast.parse(source)
        names = ('replace_sys_prompt', 'replace_sys_prompt_batch', 'update_attention_mask', 'update_attention_mask_batch')
        body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<official-replacement>', 'exec'), ns)
        tokenizer = Mock(return_value=SimpleNamespace(input_ids=MiniTensor([[1, 10, 11]])))
        tokenizer.bos_token_id = 1
        tokenizer.decode.side_effect = lambda tensor: str(tensor.values)
        original = MiniTensor([[1, 10, 11, 20, 21]])
        captured = {}
        def generate(inputs, mask, settings, embedded):
            captured.update(inputs=inputs.values, mask=mask.values, settings=settings, embedded=embedded)
            values = [[[99]]] if embedded else [[inputs.values[0] + [99]]]
            return MiniTensor(values)
        wrapper = PromptObfuscationWrapper.__new__(PromptObfuscationWrapper)
        wrapper.sampler = SimpleNamespace(tokenizer=tokenizer, device='cpu', model=SimpleNamespace(
            generation_config=SimpleNamespace(do_sample=False, remove_invalid_values=False)))
        wrapper.utils = SimpleNamespace(**{name: ns[name] for name in names})
        wrapper.backend = SimpleNamespace(generate_output=generate,
            get_embeddings=Mock(side_effect=lambda ids: MiniTensor([[[x, -x] for x in ids.values[0]]])))
        wrapper.prompt = lambda context: MiniTensor([[30, -30]] if soft else [30])
        with patch.dict(sys.modules, {'torch': fake_torch}):
            result = wrapper.generate('secret', 'AQ', {'input_ids': original, 'max_length': 60,
                'num_beams': 3, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.6})
        self.assertEqual(result, '[99]')
        self.assertEqual(original.values, [[1, 10, 11, 20, 21]])
        self.assertEqual(captured['settings']['max_new_tokens'], 55)
        self.assertNotIn('max_length', captured['settings'])
        self.assertEqual(captured['settings']['num_beams'], 3)
        self.assertEqual(captured['mask'], [[1, 1, 1, 1]])
        if soft:
            self.assertEqual(captured['inputs'], [[[1, -1], [30, -30], [20, -20], [21, -21]]])
            wrapper.backend.get_embeddings.assert_called_once()
        else:
            self.assertEqual(captured['inputs'], [[1, 30, 20, 21]])
            wrapper.backend.get_embeddings.assert_not_called()
        return wrapper, fake_torch

    def test_hard_tensor_replaces_only_context_and_keeps_budget(self):
        self.run_case(False)

    def test_soft_tensor_stays_embedded_and_is_not_decoded_as_prompt(self):
        self.run_case(True)

    def test_ambiguous_token_boundary_is_rejected(self):
        wrapper, fake_torch = self.run_case(False)
        with patch.dict(sys.modules, {'torch': fake_torch}), self.assertRaises(ValueError):
            wrapper.generate('secret', 'AQ', {'input_ids': MiniTensor([[1, 10, 999, 20]]), 'max_length': 60})


if __name__ == '__main__':
    unittest.main()
