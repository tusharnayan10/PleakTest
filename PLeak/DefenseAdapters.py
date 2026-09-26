"""Defense-only bridges. Neither attack optimization nor metrics live here."""
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import importlib
import json
import logging
import math
from pathlib import Path
import pickle
import sys
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_LOCK = RLock()  # Upstream PromptKeeper uses process-global backend bindings.


def context_key(text):
    return sha256(text.encode('utf-8')).hexdigest()


class Artifacts:
    def __init__(self, filename, model_id):
        if not filename:
            raise ValueError('This defense requires --defense-config')
        self.path = Path(filename).resolve()
        self.data = json.loads(self.path.read_text())
        if self.data['model'] != model_id:
            raise ValueError('Defense artifacts must match the PLeak target model')

    def entry(self, context):
        key = context_key(context)
        if key not in self.data['contexts']:
            raise ValueError(f'Missing defense artifact for context SHA256 {key}')
        return self.data['contexts'][key]

    def resolve(self, path):
        return (self.path.parent / path).resolve()


class PromptObfuscationWrapper:
    def __init__(self, sampler, config):
        self.sampler = sampler
        self.artifacts = Artifacts(config, sampler.model_id)
        sys.path.insert(0, str(ROOT / 'prompt_obfuscation'))
        loader_module = importlib.import_module('generate_output')
        # Upstream initializes this global only in its CLI entry point.
        loader_module.logger = logging.getLogger(loader_module.__name__)
        self.loader = loader_module.get_sys_prompt
        self.utils = importlib.import_module('src.prompt_utils')
        Model = importlib.import_module('src.model').Model
        # Bind official embedding/generation methods to PLeak's existing model.
        # Do not run upstream's loader: it changes quantization and vocabulary.
        self.backend = Model.__new__(Model)
        self.backend.model = sampler.model
        self.backend.tokenizer = sampler.tokenizer
        self.cache = {}

    def prompt(self, context):
        import torch
        key = context_key(context)
        if key not in self.cache:
            entry = self.artifacts.entry(context)
            params = json.loads(self.artifacts.resolve(entry['params']).read_text())
            if params['model_name'] != self.sampler.model_id:
                raise ValueError('Obfuscation params model does not match victim')
            saved = params['system_prompt']
            if saved.startswith('<|pad|>') and saved.endswith('<|pad|>'):
                saved = saved[len('<|pad|>'):-len('<|pad|>')]
            if saved != context:
                raise ValueError('Obfuscation was trained for a different system prompt')
            tensor = self.loader(params, False, None,
                                 str(self.artifacts.resolve(entry['tensor'])), False, '<|pad|>')
            tensor = tensor.detach().cpu()
            if tensor.ndim not in (1, 2) or tensor.shape[0] == 0:
                raise ValueError('Expected a nonempty hard [tokens] or soft [tokens, hidden] tensor')
            if tensor.ndim == 1:
                if tensor.is_floating_point() or tensor.min() < 0 or tensor.max() >= self.sampler.model.get_input_embeddings().weight.shape[0]:
                    raise ValueError('Hard prompt token IDs are incompatible with victim vocabulary')
            elif tensor.shape[1] != self.sampler.model.get_input_embeddings().weight.shape[1] or not torch.isfinite(tensor).all():
                raise ValueError('Soft prompt embeddings are incompatible with victim')
            self.cache[key] = tensor
        return self.cache[key]

    def generate(self, context, query, kwargs):
        import torch
        tokenizer = self.sampler.tokenizer
        ids = kwargs['input_ids'].cpu()
        # Keep the exact PLeak tokenization. Refuse ambiguous BPE boundaries.
        prefix = tokenizer(context, return_tensors='pt').input_ids[0]
        if not torch.equal(ids[0, :len(prefix)], prefix):
            raise ValueError('Context/query token boundary merges; cannot replace context without changing AQ tokens')
        start = 1 if len(prefix) and prefix[0].item() == tokenizer.bos_token_id else 0
        indices = [(start, len(prefix))]
        tensor = self.prompt(context)
        soft = tensor.ndim == 2
        inputs = self.backend.get_embeddings(ids.to(self.sampler.device)) if soft else ids
        if soft:
            tensor = tensor.to(inputs.dtype)
        inputs = self.utils.replace_sys_prompt_batch(tensor, inputs, indices)
        mask = self.utils.update_attention_mask_batch(len(tensor), torch.ones_like(ids), indices)
        settings = {k: v for k, v in kwargs.items() if k not in ('input_ids', 'max_length')}
        # Preserve PLeak's original response token allowance when prefix length changes.
        settings['max_new_tokens'] = kwargs['max_length'] - ids.shape[1]
        settings.setdefault('do_sample', self.sampler.model.generation_config.do_sample)
        settings['remove_invalid_values'] = self.sampler.model.generation_config.remove_invalid_values
        outputs = self.backend.generate_output(inputs.to(self.sampler.device),
                                              mask.to(self.sampler.device), settings, soft)
        tokens = outputs[0, 0] if soft else outputs[0, 0, inputs.shape[1]:]
        return tokenizer.decode(tokens)


class PromptKeeperWrapper:
    def __init__(self, sampler, config):
        self.sampler = sampler
        self.artifacts = Artifacts(config, sampler.model_id) if config else None
        sys.path.insert(0, str(ROOT / 'PromptKeeper'))
        self.defense = importlib.import_module('promptkeeper.defense')
        self.hypothesis = importlib.import_module('promptkeeper.hypothesis_test')
        self.llm = importlib.import_module('promptkeeper.llm_related')
        self.likelihood = importlib.import_module('promptkeeper.likelihood_test')
        self.thresholds = {}

    def generate_text(self, model_id, user_query, system_prompt=None, generate_kwargs=None):
        if model_id != self.sampler.model_id or not isinstance(user_query, str):
            raise ValueError('Unsupported PromptKeeper backend request')
        text = (system_prompt or '') + user_query
        tokens = self.sampler.tokenizer(text, return_tensors='pt').to(self.sampler.device)
        options = dict(generate_kwargs or self.sampler.generation_kwargs())
        options.pop('input_ids', None)
        options.pop('max_length', None)
        options.setdefault('max_new_tokens', tokens.input_ids.shape[1] + 50)
        output = self.sampler.model.generate(input_ids=tokens.input_ids, **options)
        return self.sampler.tokenizer.decode(output[0, tokens.input_ids.shape[1]:])

    @contextmanager
    def backend(self):
        # Keep official likelihood computation, fitting, decision, and mitigation.
        # Only substitute model access and serialization with the PLeak backend.
        with _LOCK, ExitStack() as stack:
            for module in (self.defense, self.hypothesis):
                stack.enter_context(patch.object(module, 'get_text_generation', self.generate_text))
            stack.enter_context(patch.object(self.hypothesis, 'get_prompt',
                lambda tokenizer, user_query, system_prompt=None: (system_prompt or '') + user_query))
            stack.enter_context(patch.object(self.hypothesis, 'AutoTokenizer',
                SimpleNamespace(from_pretrained=lambda *a, **k: self.sampler.tokenizer)))
            stack.enter_context(patch.object(self.llm, '_transformers_cached_model_id', self.sampler.model_id))
            stack.enter_context(patch.object(self.llm, '_transformers_cached_model', self.sampler.model))
            yield

    def threshold(self, model_name, system_prompt, defense_mode_params, significance):
        key = (context_key(system_prompt), significance)
        if key not in self.thresholds:
            entry = self.artifacts.entry(system_prompt)
            if entry.get('format') != 'pleak-concat-v1':
                raise ValueError('Fit must be prepared with the PLeak backend (pleak-concat-v1)')
            # Upstream writes pickle; load only locally prepared/trusted fit files.
            with self.artifacts.resolve(entry['fit']).open('rb') as stream:
                stats = pickle.load(stream)
            zero, other = stats['zero']['fitted_params'], stats['other']['fitted_params']
            if any(not math.isfinite(float(x)) for x in list(zero) + list(other)) or zero[1] <= 0 or other[1] <= 0 or zero[1] == other[1]:
                raise ValueError('Invalid or degenerate PromptKeeper calibration; prepare more calibration samples')
            threshold = self.likelihood.get_likelihood_ratio_given_significance(
                stats['dist'], list(zero) + list(other), significance)
            if not math.isfinite(float(threshold)) or threshold <= 0:
                raise ValueError('Invalid PromptKeeper threshold')
            self.thresholds[key] = (threshold, stats['dist'], zero, other)
        return self.thresholds[key]

    def generate(self, context, query, kwargs):
        from munch import DefaultMunch
        significance = self.artifacts.data.get('significance', 0.05)
        if not 0 < significance < 1:
            raise ValueError('PromptKeeper significance must be between zero and one')
        params = DefaultMunch.fromDict({'name': 'mll_test_regen',
            'hypothesis_test_significance': significance, 'no_regen': False})
        options = {k: v for k, v in kwargs.items() if k not in ('input_ids', 'max_length')}
        options['max_new_tokens'] = kwargs['max_length'] - kwargs['input_ids'].shape[1]
        with self.backend(), patch.object(self.defense, 'get_hypothesis_threshold_for_mll_test_regen', self.threshold):
            response, _ = self.defense.get_defended_model_response(
                self.sampler.model_id, params, context, query, generate_kwargs=options)
        return response


def make_wrapper(sampler, defense, config):
    if defense == 'PromptObfuscation':
        return PromptObfuscationWrapper(sampler, config)
    if defense == 'PromptKeeper':
        if not config:
            raise ValueError('PromptKeeper requires --defense-config')
        return PromptKeeperWrapper(sampler, config)
    if defense not in ('None', 'Filter'):
        raise ValueError(f'Unsupported defense {defense!r}; this PLeak checkout has no Sandwich implementation')
    return None
