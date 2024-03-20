import argparse
from collections import defaultdict
import datetime
import functools
import itertools
import os
import subprocess
import sys
import uuid

import numpy as np
import torch
from simple_slurm import Slurm
from tabulate import tabulate

from big_rl.generic.script import main as train_generic, Callbacks
from big_rl.generic.script import init_arg_parser as init_train_arg_parser
from big_rl.generic.evaluate_model import main as eval_generic
from big_rl.generic.evaluate_model import init_arg_parser as init_eval_arg_parser
from big_rl_experiments.exp1v2.common import ResultsDir, TOTAL_NUM_TASKS


def run_train(args: list[str]):
    parser = init_train_arg_parser()
    args_ns = parser.parse_args(args)
    #print(' ', args_ns.env_config)
    train_generic(args_ns)


##################################################
# Main
##################################################


def main(argv):
    #parser = init_arg_parser()
    #args = parser.parse_args()

    action = argv[0]
    argv = argv[1:]
    if action == 'train':
        run_train(argv)
    elif action == 'eval':
        #run_eval(argv)
        ...
    elif action == 'plot':
        #run_plot(argv)
        ...
    else:
        valid_actions = ['train', 'eval', 'plot']
        raise ValueError(f'Unknown action {action}. Valid actions are {valid_actions}.')


if __name__ == '__main__':
    main(sys.argv[1:])

