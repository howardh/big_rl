import os
from typing import List, Tuple, Literal
from collections import defaultdict
import logging
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import wandb


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


def get_wandb_history(run, cache_dir=None):
    """ Get the history of a wandb run, either from cache or from wandb.
    The data is cached in line-delimited json format.
    """

    if  cache_dir is None:
        for row in run.scan_history():
            yield row
        return

    cache_path = os.path.join(cache_dir, f'{run.entity}__{run.project}__{run.name}.jsonl')
    min_step = 0
    if os.path.exists(cache_path):
        log.info(f'Loading history from cache: {cache_path}')
        with open(cache_path, 'r') as f:
            for line in f.readlines():
                x = json.loads(line)
                yield x
                min_step = int(x['_step'])
    log.info(f'Loading history from wandb: {run.name}')
    history = run.scan_history(min_step=min_step)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'a') as f:
        for row in history:
            if row['_step'] == min_step and min_step > 0:
                # If `min_step` is set, it means we've already loaded data from the cache
                # If we set `min_step` when loading from wandb, it will start with the step that is equal to `min_step`
                # This will be a duplicate, so we skip it.
                # XXX: I think this won't work if we saved exactly one step to the cache at _step=0
                continue
            f.write(json.dumps(row) + '\n')
            yield row


def get_data_by_index(run, cache_dir=None):
    keys = list(filter(
            lambda k: k.startswith('env_step/by_index/') or k.startswith('reward/by_index/'),
            run.summary.keys(),
    ))
    #history = run.scan_history(page_size=100, keys=keys)
    #history = run.scan_history(keys=keys)
    #history = run.scan_history(min_step=100_000)
    #history = run.scan_history()
    history = get_wandb_history(run, cache_dir=cache_dir)

    data = defaultdict(list)
    last_obj = None
    step_key = None
    reward_key = None
    try:
        with logging_redirect_tqdm():
            for i,row in tqdm(enumerate(history)):
                #if i > 100:
                #    break
                if row.get(step_key) is None:
                    #print(step_key)
                    for k in row.keys():
                        if row.get(k) is None:
                            continue
                        if k.startswith('env_step/by_index/'):
                            last_obj = k.split('/')[-1]
                            step_key = f'env_step/by_index/{last_obj}'
                            reward_key = f'reward/by_index/{last_obj}'
                            break
                if step_key not in row:
                    raise ValueError(f'No {step_key} in {row}')
                if reward_key in row:
                    data[last_obj].append([row[step_key], row[reward_key]])
    except KeyboardInterrupt:
        pass

    return data


def get_data_by_index2(run):
    history = run.history(pandas=False)

    data = defaultdict(list)
    last_obj = None
    step_key = None
    reward_key = None
    for row in history:
        if row.get(step_key) is None:
            #print(step_key)
            for k in row.keys():
                if row.get(k) is None:
                    continue
                if k.startswith('env_step/by_index/'):
                    last_obj = k.split('/')[-1]
                    step_key = f'env_step/by_index/{last_obj}'
                    reward_key = f'reward/by_index/{last_obj}'
                    break
        if step_key not in row:
            raise ValueError(f'No {step_key} in {row}')
        if reward_key in row:
            data[last_obj].append([row[step_key], row[reward_key]])

    return data


def get_data_by_env_label(run):
    history = run.history(pandas=False)

    data = defaultdict(list)
    last_obj = None
    step_key = None
    reward_key = None
    for row in history:
        if row.get(step_key) is None:
            #print(step_key)
            for k in row.keys():
                if row.get(k) is None:
                    continue
                if k.startswith('env_step/') and not k.startswith('env_steps/by_index/'):
                    last_obj = k.split('/')[1]
                    step_key = f'env_step/{last_obj}'
                    reward_key = f'reward/{last_obj}'
                    break
        if step_key not in row:
            raise ValueError(f'No {step_key} in {row}')
        if reward_key in row:
            data[last_obj].append([row[step_key], row[reward_key]])

    return data


def linear(p1,p2,x):
    x1,y1 = p1
    x2,y2 = p2
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def resample_data(data: List[Tuple[float,float]], x: List[float], extrapolation: Literal['linear', 'constant', 'none'] = 'constant'):
    # Resample and linearly interpolate between data points
    resampled_data = []
    xi = 0
    di = 0
    while xi < len(x) and di < len(data):
        if x[xi] <= data[di][0]:
            if di == 0:
                if extrapolation == 'linear':
                    resampled_data.append((x[xi],linear(data[di], data[di+1], x[xi])))
                elif extrapolation == 'constant':
                    resampled_data.append((x[xi],data[di][1]))
                elif extrapolation == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown extrapolation method: {extrapolation}')
            else:
                # Linearly interpolate
                resampled_data.append((x[xi],linear(data[di-1], data[di], x[xi])))
            xi += 1
        else:
            if di == len(data) - 1:
                if extrapolation == 'linear':
                    resampled_data.append((x[xi],linear(data[di-1], data[di], x[xi])))
                elif extrapolation == 'constant':
                    resampled_data.append((x[xi],data[di][1]))
                elif extrapolation == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown extrapolation method: {extrapolation}')
                xi += 1
            else:
                di += 1
    return resampled_data


def plot_ema(x, y, *args, **kwargs):
    alpha = kwargs.pop('alpha', 0.9)
    y = np.array(y)
    x = np.array(x)
    y_ema = np.zeros_like(y)
    for i in range(len(y)):
        y_ema[i] = y[i] if i == 0 else alpha*y_ema[i-1] + (1-alpha)*y[i]
    return plt.plot(x, y_ema, *args, **kwargs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=str, nargs='+',
                        help='List of runs to plot in the form of "<entity>/<project>/<run>".')
    parser.add_argument('--output', type=str, default='plot.png')
    parser.add_argument('--steps-per-task', type=int, default=None)
    parser.add_argument('--ema', type=float, default=None)

    args = parser.parse_args()

    api = wandb.Api()

    run_ids = args.runs

    # Gather data
    log.info('Gathering data')
    data = {}
    for run_id in run_ids:
        try:
            log.info(f'Getting data for {run_id}')
            run = api.run(run_id)
            data[run_id] = get_data_by_index(run, cache_dir='./_cache')
        except:
            log.info(f'Failed to get data for {run_id}')
            run_ids.remove(run_id)
            raise

    # Resample so they have the same x values
    log.info('Resampling data')
    resampled_data = {}
    all_x = set()
    for run_id in run_ids:
        log.info(f'Resampling data for {run_id}')
        for obj in data[run_id].keys():
            all_x.update([x for x,_ in data[run_id][obj]])
    all_x = sorted(list(all_x))
    if len(all_x) > 1000:
        log.info(f'Too many x values ({len(all_x)}), downsample to 1000')
        all_x = all_x[::len(all_x)//1000]
    for run_id in run_ids:
        resampled_data[run_id] = {}
        for obj in data[run_id].keys():
            resampled_data[run_id][obj] = resample_data(data[run_id][obj], all_x, extrapolation='constant')
            assert len(resampled_data[run_id][obj]) == len(all_x)

    # Compute mean and std
    log.info('Computing mean and std')
    mean_data = {}
    std_data = {}
    for obj in resampled_data[run_ids[0]].keys():
        log.info(f'Computing mean and std for object {obj}')
        mean_data[obj] = []
        std_data[obj] = []
        for i in range(len(resampled_data[run_ids[0]][obj])):
            x = resampled_data[run_ids[0]][obj][i][0]
            ys = [resampled_data[run_id][obj][i][1] for run_id in run_ids]
            mean_data[obj].append((x, np.mean(ys)))
            std_data[obj].append((x, np.std(ys)))

    # Plot
    log.info('Plotting')
    for k, v in mean_data.items():
        if args.ema is not None:
            plot_ema(*zip(*v), label=k, alpha=args.ema)
        else:
            plt.plot(*zip(*v), label=k)
    # Draw a vertical line at the end of each task
    if args.steps_per_task is not None:
        for x in range(0, int(max(all_x)+1), args.steps_per_task):
            plt.axvline(x=x, color='k', linestyle='--')
    plt.legend()
    plt.grid()
    plt.xlabel('env_steps')
    plt.ylabel('reward')
    #plt.ylim(0,100)
    if args.output is not None:
        plt.savefig(args.output)
        print(f'Saved plot to {os.path.abspath(args.output)}')
    plt.close()

