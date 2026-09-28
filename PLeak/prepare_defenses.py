"""Offline artifact preparation using the bundled official defense implementations.
Run from PLeak/. This is separate from the fixed 1000-context evaluation.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

from DefenseAdapters import ROOT, context_key


def run_artifact_stage(command, expected_files):
    subprocess.run(command, cwd=ROOT / 'prompt_obfuscation', check=True)
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{command[1]} finished without required artifacts: {', '.join(missing)}. "
            "The upstream script may have logged an error and exited successfully. "
            "Check the first ERROR/traceback above, fix it, and rerun preparation.")


def validate_obfuscation_context(params, context):
    # Official --task_hints moves task instructions into training user queries,
    # leaving only the original PLeak context between the pad delimiters.
    if params.get('system_prompt') != f'<|pad|>{context}<|pad|>':
        raise ValueError(
            'Existing obfuscation artifact has a different system prompt (possibly '
            'an added task instruction). Use a fresh --output-dir to retrain it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('model')
    parser.add_argument('defense', choices=['PromptObfuscation', 'PromptKeeper'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--training-size', type=int,
                        help='Reproduce main.py dataset RNG consumption for this shadow dataset size')
    parser.add_argument('--method', choices=['hard', 'soft'], default='soft')
    parser.add_argument('--obfuscation-dataset', default='truthfulqa')
    parser.add_argument('--obfuscation-dataset-size', type=int, default=800)
    parser.add_argument('--significance', type=float, default=0.05)
    parser.add_argument('--calibration-queries', type=int, default=10)
    parser.add_argument('--calibration-repeats', type=int, default=1)
    parser.add_argument('--calibration-query-tokens', type=int, default=1024,
                        help='Offline question-list generation allowance; does not change attack budget')
    parser.add_argument('--limit', type=int, help='Prepare only first N contexts for smoke tests, not full evaluation')
    args = parser.parse_args()
    if args.calibration_queries < 2 or args.calibration_repeats < 1 or args.calibration_query_tokens < 1:
        parser.error('Use at least 2 calibration queries, 1 repeat, and a positive query token allowance')
    import numpy as np
    import torch
    from DataFactory import DataFactory
    from ModelFactory import ModelFactory
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    factory = DataFactory()
    if args.training_size is not None:
        factory.get_dataset(args.dataset, train=True, num=args.training_size)
    testset = factory.get_dataset(args.dataset, train=False, num=1000)
    # Materialize before fitting/generation consumes randomness.
    contexts = list(testset)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        contexts = contexts[:args.limit]
    model_id = ModelFactory().MODEL_CONF[args.model]['alias']
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = {'model': model_id, 'significance': args.significance, 'defense': args.defense, 'contexts': {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('defense') != args.defense:
            parser.error('Existing manifest belongs to a different defense; use a new directory')
        if manifest['model'] != model_id:
            parser.error('Existing manifest belongs to a different model')
        manifest['significance'] = args.significance
    if args.defense == 'PromptKeeper':
        from Sampler import Sampler
        from DefenseAdapters import PromptKeeperWrapper
        from munch import DefaultMunch
        sampler = Sampler(target_model=args.model, template=testset.template)
        wrapper = PromptKeeperWrapper(sampler, None)
    else:
        # Official obfuscation's system-span finder supports only these models.
        # Never silently substitute a different victim model.
        if model_id != 'meta-llama/Llama-2-7b-chat-hf':
            parser.error('Among current PLeak aliases, official obfuscation preparation supports llama-chat only')
    for context in dict.fromkeys(contexts):
        key = context_key(context)
        directory = output / key
        directory.mkdir(exist_ok=True)
        if args.defense == 'PromptObfuscation':
            params_path, tensor_path = directory / 'params.json', directory / 'best_candidate.pt'
            candidates_path = directory / 'obfuscated_system_prompt_list.pt'
            if not params_path.is_file() or not candidates_path.is_file():
                run_artifact_stage([sys.executable, 'obfuscate.py', '--model_name', model_id,
                    '--system_prompt', context, '--task_hints', '--obfuscation_method', args.method,
                    '--dataset_name', args.obfuscation_dataset,
                    '--dataset_size', str(args.obfuscation_dataset_size),
                    '--output_dir', str(directory)], [params_path, candidates_path])
            params = json.loads(params_path.read_text())
            validate_obfuscation_context(params, context)
            if params['model_name'] != model_id or params['obfuscation_method'] != args.method:
                raise ValueError('Existing obfuscation run has a different model/method; use a new directory')
            if not tensor_path.exists():
                run_artifact_stage([sys.executable, 'evaluate_obfuscation.py', '--results_dir', str(directory)],
                                   [tensor_path])
            manifest['contexts'][key] = {'params': str(params_path.relative_to(output)),
                                         'tensor': str(tensor_path.relative_to(output))}
        else:
            config = DefaultMunch.fromDict({'model': model_id,
                'num_responses_per_query': args.calibration_repeats,
                'num_queries_per_case': args.calibration_queries, 'dist': 'norm'})
            with wrapper.backend(), torch.no_grad():
                question_options = sampler.generation_kwargs()
                question_options['max_new_tokens'] = args.calibration_query_tokens
                wrapper.hypothesis.fit_distribution(
                    config, context, str(directory), question_generate_kwargs=question_options)
                wrapper.hypothesis.plt.close('all')
            manifest['contexts'][key] = {'fit': str((directory / 'fit_result.pkl').relative_to(output)),
                                         'format': 'pleak-concat-v1',
                                         'expected_samples': args.calibration_queries * args.calibration_repeats}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Artifacts: {manifest_path}')


if __name__ == '__main__':
    main()
