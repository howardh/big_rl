import argparse
from collections import defaultdict
import glob
import itertools
import logging
import os
from os.path import isfile
import sys
import yaml

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from tqdm import tqdm

from big_rl.model import factory as model_factory
from big_rl.utils.make_env import make_env_from_yaml
from big_rl.generic.script import get_action_dist_function
from big_rl.generic.evaluate_model import Callbacks as EvalCallbacks, main as eval_main, init_arg_parser as eval_init_arg_parser


logger = logging.getLogger(__name__)

EXP_NAME = 'exp3'


def plot(losses, filename):
    # Plot loss wrt steps within an episode
    colors = list(mcd.TABLEAU_COLORS.values())
    plt.figure()
    num_core_modules = len(losses)
    for m in range(num_core_modules):
        num_splits = len(losses[m])
        for i in range(num_splits):
            plt.plot(
                losses[m][i],
                c=colors[m],
                alpha=0.03,
            )
    for m in range(num_core_modules):
        plt.plot(
            # axis 2 = number of episodes, axis 0 = train-val split
            np.array(losses[m]).mean(axis=2).mean(axis=0),
            label=f'Module {m}',
            c=colors[m],
        )
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    print(f'Plot saved to {os.path.abspath(filename)}')
    plt.close()


##################################################
# Main script
##################################################


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    return parser


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    results = defaultdict(lambda: defaultdict(list)) # results[name][module #]
    for filename in glob.iglob(os.path.join(args.results_dir, 'analysis', '**', 'results.pt'), recursive=True):
        file_data = torch.load(filename)
        key = '/'.join(filename.split('/')[-3:-1])
        name, _ = key.split('/')
        num_modules = len(file_data['losses_over_time'])
        for m in range(num_modules):
            results[name][m].extend(file_data['losses_over_time'][m])

    os.makedirs(args.output_dir, exist_ok=True)
    for name, losses in results.items():
        filename = os.path.join(args.output_dir, f'plot_{name}.png')
        plot(losses, filename)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()

    main(args)


