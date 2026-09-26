"""Preserve positional CLIs and benchmark defaults; add defense configuration."""
import argparse
from hashlib import sha256
from pathlib import Path


def parse_args(training=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    if training:
        parser.add_argument('token_length', type=int)
        parser.add_argument('shadow_model')
    parser.add_argument('model')
    choices = ['None', 'Filter', 'PromptObfuscation', 'PromptKeeper']
    if training:
        parser.add_argument('train_num', type=int)
        parser.add_argument('--defense', default='None', choices=choices)
    else:
        parser.add_argument('defense', choices=choices)
        parser.add_argument('triggers')
    parser.add_argument('--defense-config', help='Per-context artifact manifest JSON')
    parser.add_argument('--output', help='CSV path (same PLeak columns)')
    args = parser.parse_args()
    if args.defense in ('PromptObfuscation', 'PromptKeeper') and not args.defense_config:
        parser.error(f'{args.defense} requires --defense-config')
    return args


def result_path(args, triggers):
    # Defense-specific folders prevent cross-defense overwrites; hashing avoids
    # treating arbitrary AQ text as a filesystem path. CSV schema is unchanged.
    path = Path(args.output) if args.output else Path('results') / args.defense / (
        f'{args.dataset}_{args.model}_{sha256(triggers.encode()).hexdigest()[:16]}.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
