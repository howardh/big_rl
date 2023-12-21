import argparse
from collections import defaultdict
import datetime
import functools
import os
import subprocess
import sys
import uuid

import numpy as np
import torch
from simple_slurm import Slurm

from big_rl.generic.script import main as train_generic
from big_rl.generic.script import init_arg_parser as init_train_arg_parser
from big_rl.generic.evaluate_model import main as eval_generic
from big_rl.generic.evaluate_model import init_arg_parser as init_eval_arg_parser


SHELL = '/bin/bash'

#GRES = 'gpu:rtx8000:1'
#GRES = 'gpu:a100l.2g.20gb:1'
GRES = 'gpu:a100l.3g.40gb:1'
#GRES = 'gpu:v100:1'


#DEBUG = False
#MODEL_CONFIG_DIR = './big_rl_experiments/exp1/configs/models'
#ENV_CONFIG_DIR = './big_rl_experiments/exp1/configs/envs'
TOTAL_NUM_TASKS = 9
#RUN_ID_PREFIX = 'exp1-2023_12_11-'
#RUN_ID_PREFIX = 'debug-'
#NUM_RANDOM_EVALS = 5 if DEBUG else 100 # Number of times to evaluate randomly initialized models for baseline performance
#MAX_STEPS = 50_000_000
#MAX_STEPS = 1_000_000

#RESULTS_DIR = os.path.join(os.environ['HOME'], 'results', 'big_rl_experiments', 'exp1', RUN_ID_PREFIX[:-1])
#CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
#EVAL_RESULTS_DIR = os.path.join(RESULTS_DIR, 'eval_results')
#EVAL_TRAINING_TASK_RESULTS_DIR = os.path.join(RESULTS_DIR, 'eval_training_task_results')
#EVAL_RANDOM_RESULTS_DIR = os.path.join(RESULTS_DIR, 'eval_random_results')
#PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

##################################################
# Training
##################################################

@functools.cache
def get_task_configs(env_config_dir: str) -> dict:
    """ Get all task config files in the provided directory.
    
    This is implemented as a cached function so that we can keep track of which tasks have been used. The 'unavailable' list is only initialized once, so it maintains all changes that are applied to it.
    """
    filenames = os.listdir(os.path.join(env_config_dir, 'train/autogenerated'))
    filenames = [f for f in filenames if f.endswith('.yaml')]
    np.random.shuffle(filenames)
    
    return {
        'all': filenames,
        'unavailable': [],
    }

def generate_training_args(num_tasks: int, exp_id, env_config_dir, model_config_dir, checkpoint_dir, max_steps, required_tasks: list[int] | None = None, debug: bool = False) -> list[str] | None:
    # Get all task config files
    #filenames = os.listdir(os.path.join(env_config_dir, 'train/autogenerated'))
    #filenames = [f for f in filenames if f.endswith('.yaml')]
    #np.random.shuffle(filenames)
    filenames = get_task_configs(env_config_dir)

    # Find the first file that has the required tasks
    env_config = None
    model_checkpoint = None
    run_id = None
    already_trained = False
    for filename in filenames['all']:
        # Count number of tasks (1s) in the filename
        num_tasks_in_filename = sum([int(c) for c in filename.split('.')[0]])
        if num_tasks_in_filename != num_tasks:
            continue
        # Check if the required tasks are in the filename
        if required_tasks is not None:
            if not all(bool(int(filename[t])) for t in required_tasks):
                continue
        # Check if the file is available
        if filename in filenames['unavailable']:
            continue
        # It is available. Reserve it.
        filenames['unavailable'].append(filename)
        # Check if the model checkpoint exists
        # If it does, it means we've already trained on this set of tasks, so we should look for another one
        model_checkpoint = os.path.join(checkpoint_dir, filename.replace('.yaml', '.pt'))
        if os.path.exists(model_checkpoint):
            model_checkpoint = None
            already_trained = True # Mark that we already trained on a set of tasks matching the criteria
            continue
        # If it doesn't exist, then we're good to go
        env_config = os.path.join(env_config_dir, 'train/autogenerated', filename)
        run_id = f'{exp_id}-{filename.replace(".yaml", "")}'
        break
    if env_config is None:
        if already_trained:
            return None
        else:
            raise ValueError('No matching config found')

    args = [
        '--env-config', env_config,
        '--model-config',
            os.path.join(model_config_dir, ('debug.yaml' if debug else 'model.yaml')),
        '--model-checkpoint', model_checkpoint,
        '--checkpoint-interval', '1_000_000',
        '--run-id', run_id,
        '--wandb-id', run_id,
        '--max-steps-total',
            ('30000' if debug else str(max_steps)),
        '--cuda',
    ]
    if not debug:
        args.append('--wandb')
    return args


def run_local(args: list[str]):
    parser = init_train_arg_parser()
    args_ns = parser.parse_args(args)
    #print(' ', args_ns.env_config)
    train_generic(args_ns)


def run_subprocess(args: list[str]):
    """ Train a model in a subprocess. This is preferable to running on the main process because the training involves logging to W&B, which doesn't like getting re-initialized in the same process. """

    script = 'big_rl/generic/script.py'
    cmd = f'python3 {script} {" ".join(args)}'

    print(cmd)

    p = subprocess.Popen(cmd, shell=True) # if shell=True, then the subprocesses will have the same environment variables as this process. Needed to pass the CUDA_VISIBLE_DEVICES variable.
    p.wait()


def run_slurm(args: list[str]):
    slurm = Slurm(
        job_name='train',
        cpus_per_task=8,
        mem='8G',
        gres=[GRES],
        output='/network/scratch/h/huanghow/slurm/%A_%a.out',

        # Run for five days
        array='1-5%1',
        time=datetime.timedelta(days=1, hours=0, minutes=0, seconds=0),

        #partition='main',
        signal='USR1@120', # Send a signal to the job 120 seconds before it is killed
    )
    #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
    slurm.add_cmd('module load python/3.10')
    slurm.add_cmd('source big_rl/ENV/bin/activate')
    slurm.add_cmd('export PYTHONUNBUFFERED=1')
    # https://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
    slurm.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
    slurm.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
    job_id = slurm.sbatch('srun python big_rl/generic/script.py ' + ' '.join(args), shell=SHELL)
    print('-'*80)
    print(slurm.script(shell=SHELL))
    print('-'*80)
    return job_id


##################################################
# Evaluation
##################################################


def generate_eval_args(model_checkpoint: str, output_dir, env_config_dir, model_config_dir, debug) -> list[str] | None:
    # Test set is the complement of the training set
    env_config = os.path.join(
        env_config_dir,
        'test/autogenerated',
        ''.join([
            '1' if c == '0' else '0' if c == '1' else c
            for c in os.path.basename(model_checkpoint)
        ]).replace('.pt', '.yaml')
    )
    print(f'{os.path.basename(model_checkpoint)} -> {env_config}')

    results = os.path.join(output_dir, os.path.basename(model_checkpoint))
    if os.path.exists(results):
        print(f'  {results} already exists')
        return None

    args = [
        '--env-config', env_config,
        '--model-config',
            os.path.join(model_config_dir, ('debug.yaml' if debug else 'model.yaml')),
        '--model', model_checkpoint,
        '--results', results,
        '--num-episodes', '10',
        '--no-video',
    ]
    return args


def generate_eval_training_task_args(model_checkpoint: str, output_dir: str, env_config_dir, model_config_dir, debug) -> list[str] | None:
    # Test models on the training tasks.
    # Get a score for normalization purposes.
    env_config = os.path.join(env_config_dir, 'test/autogenerated', os.path.basename(model_checkpoint).replace('.pt', '.yaml'))

    results = os.path.join(output_dir, os.path.basename(model_checkpoint))
    if os.path.exists(results):
        print(f'  {results} already exists')
        return None

    args = [
        '--env-config', env_config,
        '--model-config',
            os.path.join(model_config_dir, ('debug.yaml' if debug else 'model.yaml')),
        '--model', model_checkpoint,
        '--results', results,
        '--num-episodes', '10',
        '--no-video',
    ]
    return args


def generate_random_args(output_dir, env_config_dir, model_config_dir, debug) -> list[str]:
    # Test randomly initialized models.
    # Get a score for normalization purposes.
    env_config_dir = os.path.join(env_config_dir, 'test/autogenerated')
    # Choose first file and replace 0s with 1s to get the set with all tasks
    env_config = os.path.join(env_config_dir, os.listdir(env_config_dir)[0].replace('0', '1'))
    # Random results file name
    results = os.path.join(output_dir, f'{uuid.uuid4()}.pt')

    args = [
        '--env-config', env_config,
        '--model-config',
            os.path.join(model_config_dir, ('debug.yaml' if debug else 'model.yaml')),
        '--results', results,
        '--num-episodes', '1',
        '--no-video',
    ]
    return args


def eval_local(args: list[str]):
    parser = init_eval_arg_parser()
    args_ns = parser.parse_args(args)
    print(' ', args_ns.env_config)
    eval_generic(args_ns)


def eval_slurm(args: list[str], after: list[int] | None = None):
    slurm = Slurm(
        job_name='eval',
        cpus_per_task=8,
        mem='8G',
        dependency=dict(afterok=':'.join(str(job_id) for job_id in after)) if after else None,
        output='/network/scratch/h/huanghow/slurm/%A_%a.out',
        time=datetime.timedelta(days=0, hours=1, minutes=0, seconds=0),
        partition='long-cpu',
    )
    #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
    slurm.add_cmd('module load python/3.10')
    slurm.add_cmd('source big_rl/ENV/bin/activate')
    slurm.add_cmd('export PYTHONUNBUFFERED=1')
    job_id = slurm.sbatch('python big_rl/generic/evaluate_model.py ' + ' '.join(args), shell=SHELL)
    print('-'*80)
    print(slurm.script(shell=SHELL))
    print('-'*80)
    return job_id


def get_random_performance(eval_random_results_dir):
    data = defaultdict(list)
    for filename in os.listdir(eval_random_results_dir):
        if not filename.endswith('.pt'):
            continue
        results = torch.load(os.path.join(eval_random_results_dir, filename))
        for task_name, result in results.items():
            data[task_name].extend(r['episode_reward'].item() for r in result)
    return {k: np.mean(v) for k, v in data.items()}


def get_single_task_performance(eval_training_task_results_dir):
    data = defaultdict(list)
    for filename in os.listdir(eval_training_task_results_dir):
        if not filename.endswith('.pt'):
            continue
        results = torch.load(os.path.join(eval_training_task_results_dir, filename))
        for task_name, result in results.items():
            data[task_name].extend(r['episode_reward'].item() for r in result)
    return {k: np.mean(v) for k, v in data.items()}


##################################################
# Plotting
##################################################


def plot_results(output_dir, eval_results_dir, eval_random_results_dir, eval_training_task_results_dir):
    data_raw = defaultdict(lambda: defaultdict(list))
    for filename in os.listdir(eval_results_dir):
        if not filename.endswith('.pt'):
            continue
        num_tasks = os.path.basename(filename).split('.')[0].count('1')
        results = torch.load(os.path.join(eval_results_dir, filename))
        for task_name, result in results.items():
            data_raw[num_tasks][task_name].extend(r['episode_reward'].item() for r in result)
        if len(data_raw) >= 8:
            break

    random_performance = get_random_performance(eval_random_results_dir)
    single_task_performance = get_single_task_performance(eval_training_task_results_dir)

    data_normalized = defaultdict(list)
    for num_tasks, data in data_raw.items():
        for task_name, d in data.items():
            min_r = random_performance[task_name]
            max_r = single_task_performance[task_name]
            data_normalized[num_tasks].extend((x - min_r) / (max_r - min_r) for x in d)

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


def plot_slurm(argv, after):
    slurm = Slurm(
        job_name='plot',
        cpus_per_task=2,
        mem='4G',
        dependency=dict(afterok=':'.join(str(job_id) for job_id in after)) if after else None,
        output='/network/scratch/h/huanghow/slurm/%A_%a.out',
        time=datetime.timedelta(days=0, hours=1, minutes=0, seconds=0),
        partition='long-cpu',
    )
    #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
    slurm.add_cmd('module load python/3.10')
    slurm.add_cmd('source big_rl/ENV/bin/activate')
    slurm.add_cmd('export PYTHONUNBUFFERED=1')
    job_id = slurm.sbatch('python big_rl_experiments/exp1/__main__.py plot ' + ' '.join(argv))
    print('-'*80)
    print(slurm.script())
    print('-'*80)
    return job_id


##################################################
# Main
##################################################


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('actions', type=str, nargs='?',
                        choices=['train', 'eval_random', 'eval_train', 'eval_test', 'plot'],
                        default=['train', 'eval_random', 'eval_train', 'eval_test', 'plot'],
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
    parser.add_argument('--model-config-dir', type=str,
                        default='./big_rl_experiments/exp1/configs/models', 
                        help='Directory containing model configs.')
    parser.add_argument('--env-config-dir', type=str,
                        default='./big_rl_experiments/exp1/configs/envs', 
                        help='Directory containing environment configs.')

    parser.add_argument('--slurm', action='store_true',
                        help='Submit job to slurm.')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode.')

    return parser


def main_slurm(args):
    results_dir = os.path.join(os.environ['HOME'], 'results', 'big_rl_experiments', 'exp1', args.exp_id)
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    eval_results_dir = os.path.join(results_dir, 'eval_results')
    eval_training_task_results_dir = os.path.join(results_dir, 'eval_training_task_results')
    eval_random_results_dir = os.path.join(results_dir, 'eval_random_results')
    plots_dir = os.path.join(results_dir, 'plots')

    job_ids = []
    if 'train' in args.actions:
        for num_tasks in args.task_counts:
            print(f'Generating args for {num_tasks} tasks')
            used_task_configs = [] # The loop below might generate the same set of training tasks multiple times. Here, we store task configs that are already assigned so this doesn't happen.
            for required_tasks in args.required_tasks:
                # Make sure that each task appears at least once in the training set
                task_args = generate_training_args(
                    num_tasks = num_tasks,
                    required_tasks = [required_tasks],
                    exp_id = args.exp_id,
                    env_config_dir = args.env_config_dir,
                    model_config_dir = args.model_config_dir,
                    checkpoint_dir = checkpoint_dir,
                    max_steps = args.max_steps,
                    debug = args.debug,
                )
                if task_args is None:
                    continue

                job_ids.append(run_slurm(task_args))

    eval_job_ids = []

    # Evaluate randomly initialized models
    if 'eval_random' in args.actions:
        # Check how many times its been evaluated
        file_count = sum(
            1 if f.endswith('.pt') else 0
            for f in os.listdir(eval_random_results_dir)
        )
        ## XXX: Change this to run everything in series.
        #for _ in range(args.num_random_evals - file_count):
        #    task_args = generate_random_args(
        #        output_dir=eval_random_results_dir,
        #        env_config_dir = args.env_config_dir,
        #        model_config_dir = args.model_config_dir,
        #        debug = args.debug,
        #    )
        #    eval_job_ids.append(eval_slurm(task_args, after=job_ids))

    # Evaluate all models that were trained on a single task
    # Test on their training task
    if 'eval_train' in args.actions:
        for filename in os.listdir(checkpoint_dir):
            if filename.count('1') != 1:
                continue
            task_args = generate_eval_training_task_args(
                model_checkpoint = os.path.join(checkpoint_dir, filename),
                output_dir = eval_training_task_results_dir,
                env_config_dir = args.env_config_dir,
                model_config_dir = args.model_config_dir,
                debug = args.debug,
            )
            if task_args is None:
                continue
            eval_job_ids.append(eval_slurm(task_args, after=job_ids))

    # Loop through all checkpoint files and evaluate them
    if 'eval_test' in args.actions:
        for filename in os.listdir(checkpoint_dir):
            if filename.count('0') == 0: # Skip models that were trained on all tasks because the test set would be empty
                continue
            task_args = generate_eval_args(
                    model_checkpoint=os.path.join(checkpoint_dir, filename),
                    output_dir = eval_results_dir,
                    env_config_dir = args.env_config_dir,
                    model_config_dir = args.model_config_dir,
                    debug = args.debug,
            )
            if task_args is None:
                continue
            eval_job_ids.append(eval_slurm(task_args))

    # Plot the results
    if 'plot' in args.actions:
        # Replace actions with only "plot"
        argv_start_idx = 0
        for argv_start_idx, argv in enumerate(sys.argv):
            if argv.startswith('--'):
                break
        # Call this script (local version) with the same arguments, but with only "plot" as the action
        plot_slurm(sys.argv[argv_start_idx:], after=eval_job_ids)


def main_local(args):
    results_dir = os.path.join(os.environ['HOME'], 'results', 'big_rl_experiments', 'exp1', args.exp_id)
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    eval_results_dir = os.path.join(results_dir, 'eval_results')
    eval_training_task_results_dir = os.path.join(results_dir, 'eval_training_task_results')
    eval_random_results_dir = os.path.join(results_dir, 'eval_random_results')
    plots_dir = os.path.join(results_dir, 'plots')

    if 'train' in args.actions:
        for num_tasks in args.task_counts:
            print(f'Generating args for {num_tasks} tasks')
            #for required_tasks in range(TOTAL_NUM_TASKS):
            for required_tasks in args.required_tasks:
                # Make sure that each task appears at least once in the training set
                task_args = generate_training_args(
                    num_tasks = num_tasks,
                    required_tasks = [required_tasks],
                    exp_id = args.exp_id,
                    env_config_dir = args.env_config_dir,
                    model_config_dir = args.model_config_dir,
                    checkpoint_dir = checkpoint_dir,
                    max_steps = args.max_steps,
                    debug = args.debug,
                )
                if task_args is None:
                    continue

                #run_local(task_args)
                run_subprocess(task_args)
                #run_slurm(task_args)

    # Evaluate randomly initialized models
    if 'eval_random' in args.actions:
        for _ in range(args.num_random_evals):
            task_args = generate_random_args(
                output_dir=eval_random_results_dir,
                env_config_dir = args.env_config_dir,
                model_config_dir = args.model_config_dir,
                debug = args.debug,
            )
            eval_local(task_args)

    # Evaluate all models that were trained on a single task
    # Test on their training task
    if 'eval_train' in args.actions:
        for filename in os.listdir(checkpoint_dir):
            if filename.count('1') != 1:
                continue
            task_args = generate_eval_training_task_args(
                model_checkpoint = os.path.join(checkpoint_dir, filename),
                output_dir = eval_training_task_results_dir,
                env_config_dir = args.env_config_dir,
                model_config_dir = args.model_config_dir,
                debug = args.debug,
            )
            if task_args is None:
                continue
            eval_local(task_args)

    # Loop through all checkpoint files and evaluate them
    if 'eval_test' in args.actions:
        for filename in os.listdir(checkpoint_dir):
            if filename.count('0') == 0: # Skip models that were trained on all tasks because the test set would be empty
                continue
            task_args = generate_eval_args(
                    model_checkpoint=os.path.join(checkpoint_dir, filename),
                    output_dir = eval_results_dir,
                    env_config_dir = args.env_config_dir,
                    model_config_dir = args.model_config_dir,
                    debug = args.debug,
            )
            if task_args is None:
                continue
            eval_local(task_args)

    # Plot the results
    if 'plot' in args.actions:
        plot_results(
            output_dir=plots_dir,
            eval_random_results_dir=eval_random_results_dir,
            eval_training_task_results_dir=eval_training_task_results_dir,
            eval_results_dir=eval_results_dir,
        )


def main():
    parser = init_arg_parser()
    args = parser.parse_args()

    if args.slurm:
        main_slurm(args)
    else:
        main_local(args)


if __name__ == '__main__':
    main()
