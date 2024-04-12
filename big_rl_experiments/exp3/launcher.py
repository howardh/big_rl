import argparse
from collections import defaultdict
import datetime
import itertools
import math
import os
import logging
import subprocess
from torch import sub
import yaml

from simple_slurm import Slurm

from big_rl_experiments.exp3.__main__ import main


logger = logging.getLogger(__name__)

EXP_NAME = 'exp3'
SHELL = '/bin/bash' # bash is needed because `trap` behaves differently in sh and the `module` command isn't available in sh.
GRES = 'gpu:a100l.2g.20gb:1'
TRAINING_STEPS = 50_000_000


def run_train(args: list[list[str]], job_name='train', slurm: bool = False, subproc: bool = False, mem_per_task=2, dependency=None, max_steps_per_job: int = 5, duration: int = 5, time_per_job=datetime.timedelta(days=1, hours=0, minutes=0, seconds=0)) -> list[int]:
    """
    Args:
        args: List of lists of arguments to pass to the training script. Each sublist is a set of arguments to pass to a single training job.
        slurm: If True, submit the jobs to slurm. If False, run the jobs locally.
        subproc: If True, run the jobs locally as subprocesses. If False, run the jobs in the current process.
        mem_per_task: Memory per task in GB. Used only if `slurm` is True.
        dependency: Job ID to depend on. Used only if `slurm` is True.
        max_steps_per_job: Maximum number of steps to train for per job. Used only if `slurm` is True.
        duration: Duration of the job in days. Used only if `slurm` is True.
    """
    script = f'big_rl_experiments/{EXP_NAME}/__main__.py'
    action = 'train'
    if slurm:
        if len(args) > max_steps_per_job:
            job_ids = []
            for i in range(0, len(args), max_steps_per_job):
                jid = run_train(
                        args[i:i+max_steps_per_job],
                        job_name=job_name,
                        slurm=slurm,
                        subproc=subproc,
                        mem_per_task=mem_per_task,
                        dependency=dependency,
                        max_steps_per_job=max_steps_per_job,
                )
                job_ids.extend(jid)
            return job_ids
        if duration < 1:
            raise ValueError('Duration must be at least 1 day')
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=2*len(args),
            mem=f'{mem_per_task*len(args)}G',
            gres=[GRES],
            output='/network/scratch/h/huanghow/slurm/%A.out',

            # Run for five days
            array=f'1-{duration}%1',
            time=time_per_job,

            #partition='main',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source big_rl/ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        srun_args = ' '.join([
                '--overlap',
                f'--gres=gpu',
                '--output', '/network/scratch/h/huanghow/slurm/%A_%s_%a.out',
        ])
        for a in args:
            cmd = f'srun {srun_args} python {script} {action} {" ".join(a)}'
            s.add_cmd(cmd + ' &')

        print('-'*80)
        print(s.script(shell=SHELL))
        print('wait')
        print('-'*80)

        job_id = s.sbatch('wait', shell=SHELL)
        return [job_id]
    elif subproc:
        for a in args:
            cmd = f'python3 {script} {action} {" ".join(a)}'

            print(cmd)

            p = subprocess.Popen(cmd, shell=True) # if shell=True, then the subprocesses will have the same environment variables as this process. Needed to pass the CUDA_VISIBLE_DEVICES variable.
            p.wait()
        return []
    else:
        for i,a in enumerate(args):
            os.environ.update({'SLURM_STEP_ID': str(i)})
            main([action, *a])
        return []


def launch(args):
    if os.uname().nodename == 'howard-pc':
        results_dir = os.path.join(os.environ['HOME'], 'tmp', 'results', 'big_rl_experiments', EXP_NAME, args.exp_id)
    else:
        results_dir = os.path.join(os.environ['HOME'], 'results', 'big_rl_experiments', EXP_NAME, args.exp_id)

    assert isinstance(args.actions, list)

    if args.debug:
        time_per_job = datetime.timedelta(days=0, hours=0, minutes=10, seconds=0)
    else:
        time_per_job = datetime.timedelta(days=1, hours=0, minutes=0, seconds=0)
    max_steps_single_task = 30_000 if args.debug else args.max_steps_single_task
    max_steps_multi_task = (30_000 if args.debug else args.max_steps_multi_task)
    job_ids_by_task = defaultdict(list)

    # Train inner models
    task_name = 'train_single_task'
    if task_name in args.actions:
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/halfcheetah_single.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'model.yaml'),
            '--optimizer', 'Adam',
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', task_name, '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', f'{{SLURM_STEP_ID}}',
            '--wandb-id', f'{args.exp_id}__{task_name}__{{RUN_ID}}',
            '--max-steps-total', str(max_steps_single_task),
            '--cuda',
            '--wandb' if args.wandb else None,
            '--wandb-project', f'big_rl_{EXP_NAME}',
            '--wandb-group', task_name,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                max_steps_per_job=5,
                duration=5,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)


    # Multitask training (tabula rasa)
    task_name = 'train_multi_task_tr'
    if task_name in args.actions:
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/halfcheetah_multi.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'model.yaml'),
            '--optimizer', 'Adam',
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', task_name, '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', f'{{SLURM_STEP_ID}}',
            '--wandb-id', f'{args.exp_id}__{task_name}__{{RUN_ID}}',
            '--max-steps-total', str(max_steps_multi_task),
            '--cuda',
            '--wandb' if args.wandb else None,
            '--wandb-project', f'big_rl_{EXP_NAME}',
            '--wandb-group', task_name,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                max_steps_per_job=5,
                duration=5,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    # Multitask training (pre-trained, no freezing)
    task_name = 'train_multi_task_pt'
    if task_name in args.actions:
        starting_models_dir = os.path.join(results_dir, 'checkpoints', 'train_single_task')
        #starting_models = [os.path.join(starting_models_dir, f'{i}.pt') for i in range(5)]
        if len(job_ids_by_task['train_single_task']) == 0:
            dependency = None
        else:
            dependency = ':'.join(['afterany'] + [str(x) for x in job_ids_by_task['train_single_task']])
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/halfcheetah_multi.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'model.yaml'),
            '--optimizer', 'Adam',
            '--starting-model', os.path.join(starting_models_dir, '{SLURM_STEP_ID}.pt'),
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', task_name, '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', f'{{SLURM_STEP_ID}}',
            '--wandb-id', f'{args.exp_id}__{task_name}__{{RUN_ID}}',
            '--max-steps-total',
                str(max_steps_multi_task + max_steps_single_task),
            '--cuda',
            '--wandb' if args.wandb else None,
            '--wandb-project', f'big_rl_{EXP_NAME}',
            '--wandb-group', task_name,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                dependency=dependency,
                max_steps_per_job=5,
                duration=5,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    # Multitask training (pre-trained, frozen core modules)
    task_name = 'train_multi_task_pt_frozen'
    if task_name in args.actions:
        starting_models_dir = os.path.join(results_dir, 'checkpoints', 'train_single_task')
        #starting_models = [os.path.join(starting_models_dir, f'{i}.pt') for i in range(5)]
        if len(job_ids_by_task['train_single_task']) == 0:
            dependency = None
        else:
            dependency = ':'.join(['afterany'] + [str(x) for x in job_ids_by_task['train_single_task']])
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/halfcheetah_multi.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'model-frozen.yaml'),
            '--optimizer', 'Adam',
            '--starting-model', os.path.join(starting_models_dir, '{SLURM_STEP_ID}.pt'),
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', task_name, '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', f'{{SLURM_STEP_ID}}',
            '--wandb-id', f'{args.exp_id}__{task_name}__{{RUN_ID}}',
            '--max-steps-total',
                str(max_steps_multi_task + max_steps_single_task),
            '--cuda',
            '--wandb' if args.wandb else None,
            '--wandb-project', f'big_rl_{EXP_NAME}',
            '--wandb-group', task_name,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                dependency=dependency,
                max_steps_per_job=5,
                duration=5,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    # TODO: Evaluation?

    print(f'Launched {sum(len(v) for v in job_ids_by_task.values())} jobs (plotting not counted)')
    for k,v in job_ids_by_task.items():
        print(f'{k}: {v}')


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('actions', type=str, nargs='*',
                        default=['train_single_task', 'train_multi_task_tr', 'train_multi_task_pt', 'train_multi_task_pt_frozen'],
                        help='')

    parser.add_argument('--exp-id', type=str, required=True,
                        help='Identifier for the current experiment set.')
    parser.add_argument('--max-steps-single-task', type=int, default=50_000_000,
                        help='Maximum number of steps to train for (applies to single task training only, but the number of steps is split between the two tasks).')
    parser.add_argument('--max-steps-multi-task', type=int, default=50_000_000,
                        help='Maximum number of steps to train for (applies to multi-task training only).')
    parser.add_argument('--num-random-evals', type=int, default=100, 
                        help='Number of times to evaluate the random policy.')
    parser.add_argument('--model-config-dir', type=str,
                        default=f'./big_rl_experiments/{EXP_NAME}/configs/models', 
                        help='Path to the model configs.')
    parser.add_argument('--env-config-dir', type=str,
                        default=f'./big_rl_experiments/{EXP_NAME}/configs/envs', 
                        help='Directory containing environment configs.')
    parser.add_argument('--checkpoints', type=str, nargs='?', default=None,
                        help='List of checkpoints to evaluate. Used for the eval_train and eval_test actions.')

    parser.add_argument('--wandb', action='store_true',
                        help='Record run with W&B.')
    parser.add_argument('--slurm', action='store_true',
                        help='Submit job to slurm.')
    parser.add_argument('--subproc', action='store_true',
                        help='Run jobs locally as subprocesses')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode.')

    return parser


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()

    launch(args)
