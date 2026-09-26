"""One-context real-model check; does not change sample.py benchmark defaults."""
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('model')
    parser.add_argument('defense', choices=['None', 'PromptObfuscation', 'PromptKeeper'])
    parser.add_argument('triggers')
    parser.add_argument('--defense-config')
    args = parser.parse_args()
    import numpy as np
    import torch
    from DataFactory import DataFactory
    from Sampler import Sampler
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    data = DataFactory().get_dataset(args.dataset, train=False, num=1000)
    sampler = Sampler(args.model, data.template, args.defense, args.defense_config)
    results = sampler.sample_sequence([data[0]], args.triggers)
    assert len(results) == 1 and results[0]['context'] == data[0]
    for metric in ('substring', 'em', 'edit', 'semantic'):
        sampler.evaluate(results, level=metric)


if __name__ == '__main__':
    main()
