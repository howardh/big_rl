import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from tqdm import tqdm

from big_rl.utils import resample_series


def get_xy_from_dir(directory):
    # Collect all file names
    # Extract step count from file name and sort by step count
    path = Path(directory)
    files = [
        (int(fn.name.split('-')[1].split('.')[0]), fn)
        for fn in path.glob('*.pt')
    ]
    files = sorted(files, key=lambda x: x[0])

    # Check if it starts at 0. If not, then shift all steps.
    if files[0][0] != 0:
        shift = files[0][0]
        files = [(step - shift, file) for step, file in files]

    # Go through each file and get the data
    data_x = []
    data_y = defaultdict(list)
    for step, file in tqdm(files):
        if step > 10_000_000:
            break

        # Load the file
        results = torch.load(file)

        # Get the x and y values
        x = step
        rewards = {
            k: np.array([r['episode_reward'] for r in v])
            for k,v in results.items()
        }

        # Save data
        for k, v in rewards.items():
            data_y[k].append(np.mean(v))
        data_x.append(x)
        
    return data_x, data_y


def average_curves(x: list[list[int]], y: list[list[float]]) -> tuple[list, list]:
    resampled_x,resampled_ys = resample_series(list(zip(x, y)), truncate=True)
    return resampled_x, np.mean(resampled_ys, axis=0)


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--directories-tr', nargs='+', type=str, help='List of directories to plot. The directories should contain files with the naming scheme "eval-{step}.pt" where step is the number of steps the model has been trained for at the time of evaluation.')
    parser.add_argument('--directories-pt-0', nargs='+', type=str)
    parser.add_argument('--directories-pt-1', nargs='+', type=str)
    parser.add_argument('--directories-pt-2', nargs='+', type=str)
    parser.add_argument('--plot-dir', type=str)

    return parser


def main(args):
    """
    Take as input a list of directories
    Plot each curve from these directories
    Group together plots using the same module
    """
    os.makedirs(args.plot_dir, exist_ok=True)
    plot_filename = Path(args.plot_dir) / 'plot.png'

    colors = list(mcd.TABLEAU_COLORS.values())
    plt.figure()

    curves = [
        (args.directories_tr, 'TR'),
        (args.directories_pt_0, 'PT-0'),
        (args.directories_pt_1, 'PT-1'),
        (args.directories_pt_2, 'PT-2'),
    ]

    averaged_curves = []
    for directories, label in curves:
        print(f'Plotting {label}')
        c = colors.pop(0)
        all_x = []
        all_y = []
        for directory in directories:
            x, y = get_xy_from_dir(directory)
            for k, v in y.items():
                plt.plot(x, v, alpha=0.3, color=c)
                all_x.append(x)
                all_y.append(v)
        averaged_curves.append((average_curves(all_x, all_y), {'color': c, 'label': label}))

    for args, kwargs in averaged_curves:
        plt.plot(*args, **kwargs)

    plt.title('Learning Curve')
    plt.xlabel('Steps')
    plt.ylabel('Mean reward')
    plt.legend()
    plt.grid()
    plt.savefig(plot_filename)
    print(f'Plot saved to {os.path.abspath(plot_filename)}')


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
