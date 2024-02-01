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


def run_eval(args: list[str]):
    parser = init_eval_arg_parser()
    args_ns = parser.parse_args(args)
    print(' ', args_ns.env_config)
    eval_generic(args_ns)


##################################################
# Plotting
##################################################

def init_plot_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--results-dir', type=str)

    return parser


def get_performance_episode(results_dir):
    if not os.path.exists(results_dir):
        return {}

    data = defaultdict(list)
    for filename in os.listdir(results_dir):
        if not filename.endswith('.pt'):
            continue
        results = torch.load(os.path.join(results_dir, filename))
        for task_name, result in results.items():
            data[task_name].extend(r['episode_reward'].item() for r in result)
    return {k: np.mean(v) for k, v in data.items()}


def get_performance_episode_by_num_tasks(results_dir, mean=False) -> dict[int, dict[str, list[float]]]:
    if not os.path.exists(results_dir):
        return {}

    data: dict[int, dict] = defaultdict(lambda: defaultdict(list)) # {num_tasks: {task_name: [performance]}}
    for filename in os.listdir(results_dir):
        if not filename.endswith('.pt'):
            continue
        num_tasks = os.path.basename(filename).split('.')[0].count('1')
        results = torch.load(os.path.join(results_dir, filename))
        for task_name, result in results.items():
            data[num_tasks][task_name].extend(r['episode_reward'].item() for r in result)
    if mean:
        data = {
            k: {
                kk: np.mean(vv)
                for kk, vv in v.items()
            } for k, v in data.items()
        }
    return data


def get_performance_over_time(results_dir):
    if not os.path.exists(results_dir):
        return {}

    data = defaultdict(list)
    for filename in os.listdir(results_dir):
        if not filename.endswith('.pt'):
            continue
        results = torch.load(os.path.join(results_dir, filename))
        for task_name, result in results.items():
            data[task_name].extend(np.concatenate(r['results']['reward']) for r in result)
    return {k: np.mean(v) for k, v in data.items()}


def performance_table(results_dir, random_performance, single_task_performance):
    eval_performance = get_performance_episode_by_num_tasks(results_dir)

    eval_performance_mean = {}
    for num_tasks, data in eval_performance.items():
        eval_performance_mean[num_tasks] = {}
        for task_name, d in data.items():
            eval_performance_mean[num_tasks][task_name] = np.mean(d)
    num_tasks = sorted(eval_performance.keys())

    RED = '\033[1;31m'
    NOCOLOR = '\033[0m'

    def get_perf(num_tasks, task_name, normalize=False):
        if task_name not in eval_performance_mean[num_tasks]:
            return 'n/a'
        perf = eval_performance_mean[num_tasks][task_name]
        normed_perf = (perf - random_performance[task_name]) / (single_task_performance[task_name] - random_performance[task_name])

        if perf < random_performance[task_name]:
            if normalize:
                return f'{RED}{normed_perf:.2f}{NOCOLOR}'
            else:
                return f'{RED}{perf:.2f}{NOCOLOR}'

        if normalize:
            return normed_perf
        else:
            return perf

    # Render performance in a table
    row_headings = sorted(set(list(random_performance.keys()) + list(single_task_performance.keys())))
    col_headings = ['', 'Random', 'Single Task'] + [f'{x} Tasks' for x in num_tasks] + [f'{x} Tasks (N)' for x in num_tasks]
    table_data = [
        [
            task_name,
            random_performance.get(task_name, 'n/a'),
            single_task_performance.get(task_name, 'n/a'),
        ] + [
            get_perf(x, task_name) for x in num_tasks
        ] + [
            get_perf(x, task_name, normalize=True) for x in num_tasks
        ]
        for task_name in row_headings
    ]
    table = tabulate(table_data, headers=col_headings, tablefmt='simple_grid')
    print(table)


def plot_performance_episode(eval_results_dir, output_dir, random_performance, single_task_performance):
    # Gather performance on the eval tasks
    # Organize by number of training tasks, and the test task
    data_raw = get_performance_episode_by_num_tasks(eval_results_dir)
    #data_raw = defaultdict(lambda: defaultdict(list)) # {num_tasks: {task_name: [performance]}}
    #for filename in os.listdir(eval_results_dir):
    #    if not filename.endswith('.pt'):
    #        continue
    #    num_tasks = os.path.basename(filename).split('.')[0].count('1')
    #    results = torch.load(os.path.join(eval_results_dir, filename))
    #    for task_name, result in results.items():
    #        data_raw[num_tasks][task_name].extend(r['episode_reward'].item() for r in result)
    #    if len(data_raw) >= 8: # XXX: ??? What is this for? I don't remember why it's here.
    #        break

    # Normalize data such that 0 is random performance and 1 is the performance of a model trained on that task alone
    data_normalized = defaultdict(list)
    for num_tasks, data in data_raw.items():
        for task_name, d in data.items():
            min_r = random_performance[task_name]
            max_r = single_task_performance[task_name]
            data_normalized[num_tasks].extend((x - min_r) / (max_r - min_r) for x in d)

    # Render generalization performance in a table
    row_headings = sorted(data_normalized.keys())
    col_headings = ['Num Training Tasks', 'Mean', 'Std', 'n']
    table_data = [[
            num_tasks,
            np.mean(data_normalized[num_tasks]),
            np.std(data_normalized[num_tasks]),
            len(data_normalized[num_tasks]),
        ] for num_tasks in row_headings
    ]
    table = tabulate(table_data, headers=col_headings, tablefmt='simple_grid')
    print('Generalization Performance')
    print(table)

    # Render Welch's t-test results in a table
    import scipy
    def welch(data1, data2):
        """ Check if data1 is smaller than data2 """
        t, p = scipy.stats.ttest_ind(data1, data2, equal_var=False, alternative='less')
        return p
    row_headings = sorted(data_normalized.keys())
    col_headings = [''] + row_headings
    table_data = [
            [n1] + 
            [welch(data_normalized[n1], data_normalized[n2]) if n2 >= n1 else '' for n2 in row_headings]
            for n1 in row_headings
    ]
    table = tabulate(table_data, headers=col_headings, tablefmt='simple_grid')
    print('Welch\'s t-test p-values (one-tailed)')
    print(table)

    # Render generalization performance as box plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = sorted(data_normalized.keys())
    data = [data_normalized[k] for k in labels]

    plt.figure()
    plt.title('Generalization performance of models trained on different numbers of tasks')
    plt.xlabel('Number of training tasks')
    plt.ylabel('Normalized performance')
    plt.grid(axis='y', which='both', linestyle='--')
    plt.boxplot(data, labels=labels)
    filename = os.path.join(output_dir, 'boxplot.png')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)

    print(f'Saved plot to {os.path.abspath(filename)}')


def plot_performance_over_time(eval_results_dir, output_dir, random_performance, single_task_performance):
    # Gather performance on the eval tasks
    data_raw = defaultdict(lambda: defaultdict(list)) # {num_tasks: {task_name: [performance]}}
    for filename in os.listdir(eval_results_dir):
        if not filename.endswith('.pt'):
            continue
        num_tasks = os.path.basename(filename).split('.')[0].count('1')
        results = torch.load(os.path.join(eval_results_dir, filename))
        for task_name, result in results.items():
            data_raw[num_tasks][task_name].extend(np.concatenate(r['results']['reward']) for r in result)

    # Normalize data such that 0 is random performance and 1 is the performance of a model trained on that task alone
    data_normalized = defaultdict(lambda: defaultdict(list))
    for num_tasks, data in data_raw.items():
        for task_name, d in data.items():
            min_r = random_performance[task_name]
            max_r = single_task_performance[task_name]
            data_normalized[num_tasks][task_name].extend((x - min_r) / (max_r - min_r) for x in d)

    # Average across episodes, trimmed to the shortest length episode
    data_avg_short = defaultdict(dict)
    for num_tasks, data in data_raw.items():
        for task_name, d in data.items():
            #max_len = max(len(x) for x in d)
            min_len = min(len(x) for x in d)
            data_avg_short[num_tasks][task_name] = np.mean(np.stack([x[:min_len] for x in d]), axis=0)

    # Average across episodes, padded to the longest length episode
    data_avg_long = defaultdict(dict)
    for num_tasks, data in data_raw.items():
        for task_name, d in data.items():
            max_len = max(len(x) for x in d)
            data_avg_long[num_tasks][task_name] = np.nanmean(np.stack([
                np.pad(x, (0, max_len - x.shape[0]), mode='constant', constant_values=np.nan)
                for x in d
            ]), axis=0)
            
    from matplotlib import pyplot as plt

    def plot_avg(data_avg, filename):
        fig, axes = plt.subplots(
                ncols=len(data_avg),
                nrows=len(data_avg[1]),
                figsize=(4*len(data_avg), 3*len(data_avg[1])),
                sharey='row', # type: ignore
        )
        # Get list of task names so that we can handle missing data
        all_task_names = set()
        for i, (num_tasks, data) in enumerate(data_avg.items()):
            for j, (task_name, y) in enumerate(sorted(data.items())):
                all_task_names.add(task_name)
        # Plot
        for i, (num_tasks, data) in enumerate(sorted(data_avg.items())):
            for j, task_name in enumerate(sorted(all_task_names)):
                ax = axes[j, i]
                all_task_names.add(task_name)

                if task_name not in data:
                    continue

                y = data[task_name]
                x = np.arange(len(y))

                ax.grid(axis='y', which='both', linestyle=':')
                ax.axhline(0, color='gray', linestyle='--')
                #ax.set_title(f'{task_name} ({num_tasks} tasks)')
                ax.plot(x, y)
        # Labels
        row_labels = sorted(all_task_names)
        col_labels = [f'{x} task(s)' for x in sorted(data_avg.keys())]
        for ax, col in zip(axes[0], col_labels):
            ax.set_title(col)
        for ax, row in zip(axes[:,0], row_labels):
            ax.set_ylabel(row, rotation=90, size='large')
        ## Make sure all plots have the same y-axis scale so we can compare them
        #for j, task_name in enumerate(sorted(all_task_names)):
        #    ylims = []
        #    for i, (num_tasks, data) in enumerate(data_avg.items()):
        #        ax = axes[j, i]
        #        ylims.append(ax.get_ylim())
        #    for i, (num_tasks, data) in enumerate(data_avg.items()):
        #        ax = axes[j, i]
        #        ax.set_ylim([min(y[0] for y in ylims), max(y[1] for y in ylims)])
        #plt.show()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        print(f'Saved plot to {os.path.abspath(os.path.join(output_dir, filename))}')
        plt.close()

    #plot_avg(data_avg_short)
    plot_avg(data_avg_long, 'performance_over_time.png')


def plot_test_vs_train_performance(eval_train_results_dir, eval_train_all_results_dir, eval_results_dir, output_dir, random_performance, single_task_performance):
    """ Produce a plot to see how well the model generalizes as a function of how well it learned the training tasks """
    data_raw = defaultdict(lambda: defaultdict(list)) # {num_tasks: [(train_performance, test_performance)]}
    for filename in os.listdir(eval_results_dir):
        if not filename.endswith('.pt'):
            continue
        num_tasks = os.path.basename(filename).split('.')[0].count('1')
        # Load test results
        test_results = torch.load(os.path.join(eval_results_dir, filename))
        # Load train results
        if num_tasks == 1:
            train_results = torch.load(os.path.join(eval_train_results_dir, filename))
        else:
            train_results = torch.load(os.path.join(eval_train_all_results_dir, filename))
        # Compute point
        def get_normalized_performance(results):
            data = {}
            for task_name, result in results.items():
                data[task_name] = np.mean([r['episode_reward'].item() for r in result])
                min_r = random_performance[task_name]
                max_r = single_task_performance[task_name]
                data[task_name] = (data[task_name] - min_r) / (max_r - min_r)
            return data
        test_performance = get_normalized_performance(test_results)
        train_performance = np.mean(list(get_normalized_performance(train_results).values()))
        for task_name, performance in test_performance.items():
            data_raw[num_tasks][task_name].append((train_performance, performance))

    # Produce scatter plot of train vs test performance
    from matplotlib import pyplot as plt
    
    plt.figure()
    plt.xlabel('Train performance')
    plt.ylabel('Test performance')
    plt.title('Test performance vs train performance')
    plt.grid(axis='both', linestyle=':')
    for num_tasks, data in sorted(data_raw.items()):
        if num_tasks == 1:
            continue
        points = list(itertools.chain(*data.values()))
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x, y, label=f'{num_tasks} tasks')
    plt.legend()
    filename = os.path.join(output_dir, f'test_vs_train_performance.png')
    plt.savefig(filename)
    print(f'Saved plot to {os.path.abspath(filename)}')
    plt.close()


def plot(output_dir, results_dir):
    eval_test_results_dir = ResultsDir.from_results_dir(os.path.join(results_dir, 'eval_test_results'))
    eval_train_results_dir = ResultsDir.from_results_dir(os.path.join(results_dir, 'eval_train_results'))
    eval_random_results_dir = os.path.join(results_dir, 'eval_random_results')

    random_performance = get_performance_episode(eval_random_results_dir)
    single_task_performance = get_performance_episode_by_num_tasks(eval_train_results_dir.default, mean=True)[1]
    #single_task_performance = get_performance_episode(eval_train_results_dir)

    result_dirs = [
        #('standard setup', eval_results_dir.default),
        #('shuffled observation', eval_results_dir.shuffled_obs),
        #('shuffled action', eval_results_dir.shuffled_action),
        #('shuffled observation and action', eval_results_dir.shuffled_obs_and_action),
        #('occluded obs', eval_results_dir.occluded_obs_100),
        #('occluded obs action and reward', eval_test_results_dir.occluded_obs_action_reward_100),

        ('standard setup (training tasks)', eval_train_results_dir.default),
        ('occluded obs action and reward (training tasks)', eval_train_results_dir.occluded_obs_action_reward_100),
    ]
    for description, result_dir in result_dirs:
        print('#'*80)
        print(f'# Test performance with {description}')
        print('#'*80)

        performance_table(result_dir, random_performance, single_task_performance)

        if not os.path.exists(result_dir):
            print(f'No results found in {eval_test_results_dir.default}')
            print('  Models have not been evaluated on test tasks yet. Skipping plotting.')
            continue

        plot_performance_episode(
            eval_results_dir=result_dir,
            output_dir=output_dir,
            random_performance=random_performance,
            single_task_performance=single_task_performance,
        )

        #plot_performance_over_time(
        #    eval_results_dir=result_dir,
        #    output_dir=output_dir,
        #    random_performance=random_performance,
        #    single_task_performance=single_task_performance,
        #)

        #plot_test_vs_train_performance(
        #    eval_train_results_dir=eval_train_results_dir,
        #    eval_train_all_results_dir=eval_train_all_results_dir,
        #    eval_results_dir=result_dir,
        #    output_dir=output_dir,
        #    random_performance=random_performance,
        #    single_task_performance=single_task_performance,
        #)


def run_plot(argv):
    parser = init_plot_arg_parser()
    args = parser.parse_args(argv)

    plot(
        output_dir = args.output_dir,
        results_dir = args.results_dir
    )


##################################################
# Main
##################################################


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str,
                        choices=['train', 'eval', 'plot'],
                        help='')

    parser.add_argument('--exp-id', type=str, default=None,
                        help='Identifier for the current experiment set.')
    parser.add_argument('--task-counts', type=int, nargs='+', 
                        default=list(range(1, TOTAL_NUM_TASKS)),
                        help='Train on sets of tasks of these sizes.')
    parser.add_argument('--required-tasks', type=int, nargs='+', 
                        default=list(range(TOTAL_NUM_TASKS)),
                        help='Train on sets of tasks that include these tasks. Each set of tasks will be guaranteed to include one of the these tasks.')
    parser.add_argument('--max-steps', type=int, default=50_000_000,
                        help='Maximum number of steps to train for.')
    parser.add_argument('--num-random-evals', type=int, default=100, 
                        help='Number of times to evaluate the random policy.')
    parser.add_argument('--model-config', type=str,
                        default='./big_rl_experiments/exp1/configs/models/model.yaml', 
                        help='Path to the model configs.')
    parser.add_argument('--env-config-dir', type=str,
                        default='./big_rl_experiments/exp1/configs/envs', 
                        help='Directory containing environment configs.')
    parser.add_argument('--checkpoints', type=str, nargs='?', default=None,
                        help='List of checkpoints to evaluate. Used for the eval_train and eval_test actions.')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode.')

    return parser


def main():
    #parser = init_arg_parser()
    #args = parser.parse_args()

    action = sys.argv[1]
    argv = sys.argv[2:]
    if action == 'train':
        run_train(argv)
    elif action == 'eval':
        run_eval(argv)
    elif action == 'plot':
        run_plot(argv)
    else:
        valid_actions = ['train', 'eval', 'plot']
        raise ValueError(f'Unknown action {action}. Valid actions are {valid_actions}.')


if __name__ == '__main__':
    main()
