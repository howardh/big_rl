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
from big_rl_experiments.exp3.plot_ball_learning_curve import main as plot_ball_main, init_arg_parser as init_plot_ball_arg_parser


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
    from big_rl.generic.evaluate_model import main as main_, init_arg_parser as init_arg_parser_

    parser = init_arg_parser_()
    args_ns = parser.parse_args(args)
    main_(args_ns)


def run_analysis(args: list[str]):
    from big_rl_experiments.exp3.eval_module_belief import main as main_, init_arg_parser as init_arg_parser_

    parser = init_arg_parser_()
    args_ns = parser.parse_args(args)
    main_(args_ns)


def run_plot_module_beliefs(args: list[str]):
    from big_rl_experiments.exp3.plot_module_belief import main as main_, init_arg_parser as init_arg_parser_

    parser = init_arg_parser_()
    args_ns = parser.parse_args(args)
    main_(args_ns)


def run_plot_ball(args: list[str]):
    """ Plot learning curves for ball task """
    from big_rl_experiments.exp3.plot_ball_learning_curve import main as main_, init_arg_parser as init_arg_parser_

    parser = init_arg_parser_()
    args_ns = parser.parse_args(args)
    main_(args_ns)


def run_archive(args: list[str]):
    import tarfile

    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('dst', type=str)
    args_ns = parser.parse_args(args)

    os.makedirs(os.path.dirname(args_ns.dst), exist_ok=True)

    # TODO: Check if tar file exists
    # TODO: If it exists, check if there's files in the archive that do not exist in the source directory. If so, it suggests that some may have been removed due to surpassing the 90 day limit. In this case, add the new files to the archive and keep the extra files where they are.

    with tarfile.open(args_ns.dst, 'w:gz') as tar:
        tar.add(args_ns.src, arcname=os.path.basename(args_ns.src))
    print(f'Archived created at {os.path.abspath(args_ns.dst)}')


##################################################
# Main
##################################################


def main(argv):
    action = argv[0]
    argv = argv[1:]
    if action == 'train':
        run_train(argv)
    elif action == 'eval':
        run_eval(argv)
    elif action == 'analysis':
        run_analysis(argv)
    elif action == 'plot_module_beliefs':
        run_plot_module_beliefs(argv)
    elif action == 'plot':
        #run_plot(argv)
        ...
    elif action == 'plot_ball_learning_curve':
        run_plot_ball(argv)
    elif action == 'archive':
        run_archive(argv)
    else:
        valid_actions = ['train', 'eval', 'analysis', 'plot', 'plot_module_beliefs', 'plot_ball_learning_curve', 'archive']
        raise ValueError(f'Unknown action {action}. Valid actions are {valid_actions}.')


if __name__ == '__main__':
    main(sys.argv[1:])

