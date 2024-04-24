import argparse
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
from tqdm import tqdm

from big_rl.model import factory as model_factory
from big_rl.utils.make_env import make_env_from_yaml
from big_rl.generic.script import get_action_dist_function
from big_rl.generic.evaluate_model import Callbacks as EvalCallbacks, main as eval_main, init_arg_parser as eval_init_arg_parser


logger = logging.getLogger(__name__)

EXP_NAME = 'exp3'

##################################################
# Belief model
##################################################


class BeliefModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


def generate_splits(n, k):
    """ Generate k splits of n elements """
    indices = torch.randperm(n)
    chunks = torch.chunk(indices, k)
    for i in range(k):
        train = torch.cat(chunks[:i] + chunks[i+1:])
        val = chunks[i]
        yield train, val


def train_belief_model(data, num_epochs=10, plot: str | None = None):
    """ Train and return score """

    # Train with LOOCV on the dataset for each module
    num_core_modules = len(data['fw'])
    num_splits = 5
    #results = [[] for _ in range(num_core_modules)]
    loss_history = [
            {
                'train': [[] for _ in range(num_splits)],
                'val': [[] for _ in range(num_splits)],
            }
            for _ in range(num_core_modules)
    ]

    for m in range(num_core_modules):
        x1 = torch.stack(data['fw'][m])
        x2 = torch.stack(data['bw'][m])
        x = torch.cat([x1,x2])
        y = torch.tensor([0.] * len(x1) + [1.] * len(x2))

        for split_idx, (train_indices, val_indices) in enumerate(generate_splits(len(x), num_splits)):
            bce = torch.nn.BCEWithLogitsLoss()

            model = torch.nn.Linear(in_features=x.shape[1], out_features=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            best_val_loss = float('inf')
            for _ in tqdm(range(num_epochs), desc=f'Module {m} split {split_idx}'):
                y_pred = model(x[train_indices])
                train_loss = bce(y_pred.squeeze(), y[train_indices])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                y_pred = model(x[val_indices])
                val_loss = bce(y_pred.squeeze(), y[val_indices])

                #best_val_loss = min(best_val_loss, val_loss.item())

                loss_history[m]['train'][split_idx].append(train_loss.item())
                loss_history[m]['val'][split_idx].append(val_loss.item())

            #results[m].append(best_val_loss)

    if plot is not None:
        _, ax = plt.subplots(1, num_core_modules, figsize=(5*num_core_modules, 5), sharey=True)
        for i in range(num_core_modules):
            ax[i].plot(
                    list(zip(*loss_history[i]['train'])),
                    c='tab:blue',
                    alpha=0.1,
            )
            ax[i].plot(
                    list(zip(*loss_history[i]['val'])),
                    c='tab:orange',
                    alpha=0.1,
            )
            ax[i].plot(
                    np.mean(loss_history[i]['train'], axis=0),
                    label='train',
                    c='tab:blue',
            )
            ax[i].plot(
                    np.mean(loss_history[i]['val'], axis=0),
                    label='val',
                    c='tab:orange',
            )
            ax[i].set_title(f'Module {i}')
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel('Loss')
            ax[i].set_yscale('log')
            ax[i].grid()
            ax[i].legend()
        plt.savefig(plot)
        plt.close()

    results = [np.mean(l['val'], axis=0).min() for l in loss_history]

    return results


##################################################
# Callbacks
##################################################


##################################################
# Main script
##################################################


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-config-dir', type=str,
                        default=f'./big_rl_experiments/{EXP_NAME}/configs/models', 
                        help='Path to the model configs.')
    parser.add_argument('--env-config-dir', type=str,
                        default=f'./big_rl_experiments/{EXP_NAME}/configs/envs', 
                        help='Directory containing environment configs.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint to evaluate.')
    parser.add_argument('--results-dir', type=str, required=True)

    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--dataset-size', type=int, default=1000)
    parser.add_argument('--cache-dataset', action='store_true',
                        help='Cache dataset to avoid recreating it. Use for debugging purposes only. When evaluating a model, we want to generate a new dataset on each run.')

    return parser


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    dataset_filename = os.path.join(args.results_dir, 'dataset.pt')
    results_filename = os.path.join(args.results_dir, 'results.pt')
    plot_filename = os.path.join(args.results_dir, 'plot.png')

    def get_dataset():
        data = None

        if os.path.isfile(dataset_filename) and args.cache_dataset:
            data = torch.load(dataset_filename)
        else:
            def on_model_init(locals_):
                nonlocal data

                model = locals_['model']

                num_core_modules = len(model.core_modules)
                data = {
                    'fw': [[] for _ in range(num_core_modules)],
                    'bw': [[] for _ in range(num_core_modules)],
                    'fw_t': [[] for _ in range(num_core_modules)],
                    'bw_t': [[] for _ in range(num_core_modules)],
                }

            def on_step_callback(locals_):
                """ Evaluate the model after each training step """
                nonlocal data
                assert data is not None

                task_name = locals_['metadata']['task_name']

                # locals_['hidden'][0] and locals_['hidden'][1] are the k/v pairs from the core modules. I don't remember which is which, but the order doesn't matter, as long as it's consistent.
                h1 = locals_['hidden'][0].squeeze(1)
                h2 = locals_['hidden'][1].squeeze(1)
                t = locals_['step']

                for i in range(len(data[task_name])):
                    if len(data[task_name][i]) >= args.dataset_size:
                        continue
                    data[task_name][i].append(torch.cat([h1[i], h2[i]]).detach().cpu())
                    data[task_name + '_t'][i].append(t)

            callbacks = EvalCallbacks(
                on_model_init = on_model_init,
                on_step = on_step_callback,
            )

            eval_parser = eval_init_arg_parser()
            eval_main(
                args = eval_parser.parse_args([
                    '--env-config',
                        os.path.join(args.env_config_dir, 'test/halfcheetah_single_100.yaml'),
                    '--model-config',
                        os.path.join(args.model_config_dir, 'model.yaml'),
                    '--model', args.checkpoint,
                    '--no-video',
                    '--num-episodes', str(args.num_episodes)
                ]),
                callbacks = callbacks,
            )

            torch.save(data, dataset_filename)
        return data

    # Train belief model on each dataset
    if isfile(results_filename):
        results = torch.load(results_filename)
    else:
        results = train_belief_model(
            get_dataset(),
            num_epochs=args.num_epochs,
            plot=plot_filename,
        )
        torch.save(results, results_filename)

    #print([np.mean(r) for r in results])
    print(results)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()

    main(args)

