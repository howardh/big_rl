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
from big_rl_experiments.exp3.eval_module_belief import main as analyse_main, init_arg_parser as init_analyse_arg_parser
from big_rl.generic.evaluate_model import main as eval_main, init_arg_parser as init_eval_arg_parser


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def on_model_init(locals_):
    ## Checking if the submodel stuff is working properly
    #model = locals_['model']
    #model_fw = model['forward']
    #model_bw = model['backward']
    #for p in model_fw.parameters():
    #    p.data.fill_(0.0)
    #params = [p.mean().item() for p in model_bw.parameters()]
    #breakpoint()
    pass


def run_train(args: list[str]):
    parser = init_train_arg_parser()
    args_ns = parser.parse_args(args)
    train_generic(args_ns, callbacks=Callbacks(on_model_init=on_model_init))


def run_eval(args: list[str]):
    parser = init_eval_arg_parser()
    args_ns = parser.parse_args(args)
    eval_main(args_ns)


def run_analysis(args: list[str]):
    parser = init_analyse_arg_parser()
    args_ns = parser.parse_args(args)
    analyse_main(args_ns)


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
        run_eval(argv)
    elif action == 'analysis':
        run_analysis(argv)
    elif action == 'plot':
        #run_plot(argv)
        ...
    else:
        valid_actions = ['train', 'eval', 'analysis', 'plot']
        raise ValueError(f'Unknown action {action}. Valid actions are {valid_actions}.')


if __name__ == '__main__':
    main(sys.argv[1:])

