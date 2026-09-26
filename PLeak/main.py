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

args = parse_args(training=True)
dataset, token_length = args.dataset, args.token_length
shadow_model, target_model, train_num = args.shadow_model, args.model, args.train_num
test_num = 1000

dataFactory = DataFactory()
trainset = dataFactory.get_dataset(dataset, train=True, num=train_num)
testset = dataFactory.get_dataset(dataset, train=False, num=test_num)
attack = HotFlip(trigger_token_length=token_length, shadow_model=shadow_model, template=trainset.template)
attack.replace_triggers(trainset)

triggers = attack.decode_triggers()

del attack
torch.cuda.empty_cache() 

sampler = Sampler(target_model=target_model, template=testset.template, defense=args.defense, defense_config=args.defense_config)
results = sampler.sample_sequence(testset, triggers=triggers)
Sampler.save_to_csv(result_path(args, triggers), results, triggers)

sampler.evaluate(results, level='substring')
sampler.evaluate(results, level='em')
sampler.evaluate(results, level='edit')
sampler.evaluate(results, level='semantic')
