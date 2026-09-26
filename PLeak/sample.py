from Attack import HotFlip
from Sampler import Sampler
import random
from datasets import load_dataset
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict
import os
from DataFactory import DataFactory
import sys
from DefenseCLI import parse_args, result_path

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

args = parse_args()
dataset, model, defense, triggers = args.dataset, args.model, args.defense, args.triggers
test_num = 1000

dataFactory = DataFactory()
testset = dataFactory.get_dataset(dataset, train=False, num=test_num)

sampler = Sampler(target_model=model, template=testset.template, defense=defense, defense_config=args.defense_config)
results = sampler.sample_sequence(testset, triggers=triggers)
Sampler.save_to_csv(result_path(args, triggers), results, triggers)

sampler.evaluate(results, level='substring')
sampler.evaluate(results, level='em')
sampler.evaluate(results, level='edit')
sampler.evaluate(results, level='semantic')
