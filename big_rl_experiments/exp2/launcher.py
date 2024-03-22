import argparse
import datetime
import itertools
import math
import os
import logging
import subprocess
from torch import sub
import yaml

from simple_slurm import Slurm

from big_rl_experiments.exp2.__main__ import main


logger = logging.getLogger(__name__)

EXP_NAME = 'exp2'
SHELL = '/bin/bash' # bash is needed because `trap` behaves differently in sh and the `module` command isn't available in sh.
GRES = 'gpu:a100l.2g.20gb:1'
TRAINING_STEPS = 50_000_000


def run_train(args: list[list[str]], job_name='train', slurm: bool = False, subproc: bool = False, mem_per_task=2, dependency=None, max_steps_per_job: int = 5, duration: int = 5) -> list[int]:
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
            time=datetime.timedelta(days=1, hours=0, minutes=0, seconds=0),

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

    train_job_ids = []
    train_inner_job_ids = []

    # Train inner models
    if 'train_inner' in args.actions:
        for direction in ['forward', 'backward']:
            print(f'Generating args for {direction} tasks')
            task_args = [
                '--env-config',
                    os.path.join(args.env_config_dir, f'train/{direction}/halfcheetah.yaml'),
                '--model-config',
                    os.path.join(args.model_config_dir, 'lstm1.yaml'),
                '--optimizer', 'Adam',
                '--model-checkpoint',
                    os.path.join(results_dir, 'checkpoints', 'inner', '{RUN_ID}.pt'),
                '--checkpoint-interval', '1_000_000',
                '--run-id', f'{direction}-{{SLURM_STEP_ID}}',
                '--wandb-id', f'{args.exp_id}_inner_{{RUN_ID}}',
                '--max-steps-total',
                    ('30000' if args.debug else str(args.max_steps_inner)),
                '--cuda',
                '--wandb' if args.wandb else None,
            ]
            task_args = [a for a in task_args if a is not None]
            job_ids = run_train(
                    [task_args] * 5,
                    job_name=f'train_inner_{direction}',
                    slurm=args.slurm,
                    subproc=args.subproc,
                    max_steps_per_job=5,
                    duration=5,
            )
            train_inner_job_ids.extend(job_ids)
            train_job_ids.extend(job_ids)
    train_inner_job_ids = [str(i) for i in train_inner_job_ids] # Convert to str so that it can be used with `str.join` later.

    # Train horizontal hierarchical model (without pre-trained inner model)
    if 'train_h_tr' in args.actions: # h=horizontal hierarchy, tr=tabula rasa
        direction = 'both'
        print(f'Generating args for horizontal hierarchy')
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/{direction}/halfcheetah.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'horizontal.yaml'),
            '--optimizer', 'Adam',
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', 'h_tr', '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', '{SLURM_STEP_ID}',
            '--wandb-id', f'{args.exp_id}_h_tr_{{RUN_ID}}',
            '--max-steps-total',
                ('30000' if args.debug else str(args.max_steps_outer)),
            '--cuda',
            '--wandb' if args.wandb else None,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name='train_h_tr',
                slurm=args.slurm,
                max_steps_per_job=5,
                duration=math.ceil(args.max_steps_outer / 15_000_000) + 2, # Run approximately 17M steps per day. Assume 15M and add 2 days as buffer.
        )
        train_job_ids.extend(job_ids)

    # Train vertical hierarchical model (without pre-trained inner model)
    if 'train_v_tr' in args.actions:
        direction = 'both'
        print(f'Generating args for vertical hierarchy')
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/{direction}/halfcheetah.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'vertical.yaml'),
            '--optimizer', 'Adam',
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', 'v_tr', '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', '{SLURM_STEP_ID}',
            '--wandb-id', f'{args.exp_id}_v_tr_{{RUN_ID}}',
            '--max-steps-total',
                ('30000' if args.debug else str(args.max_steps_outer)),
            '--cuda',
            '--wandb' if args.wandb else None,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name='train_v_tr',
                slurm=args.slurm,
                max_steps_per_job=5,
                duration=math.ceil(args.max_steps_outer / 15_000_000) + 2, # Run approximately 17M steps per day. Assume 15M and add 2 days as buffer.
        )
        train_job_ids.extend(job_ids)

    # Train size-equivalent LSTM
    if 'train_lstm' in args.actions:
        direction = 'both'
        print(f'Generating args for {direction} tasks')
        task_args = [
            '--env-config',
                os.path.join(args.env_config_dir, f'train/{direction}/halfcheetah.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'lstm2.yaml'),
            '--optimizer', 'Adam',
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', 'lstm', '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--run-id', '{SLURM_STEP_ID}',
            '--wandb-id', f'{args.exp_id}_lstm_{{RUN_ID}}',
            '--max-steps-total',
                ('30000' if args.debug else str(args.max_steps_outer)),
            '--cuda',
            '--wandb' if args.wandb else None,
        ]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                [task_args] * 5,
                job_name='train_lstm',
                slurm=args.slurm,
                max_steps_per_job=5,
                duration=math.ceil(args.max_steps_outer / 25_000_000) + 2, # Run approximately 27M steps per day. Assume 25M and add 2 days as buffer.
        )
        train_job_ids.extend(job_ids)

    if 'train_h_pt' in args.actions:
        # Create model config for next step
        with open(os.path.join(args.model_config_dir, 'horizontal.yaml'), 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        os.makedirs(os.path.join(results_dir, 'model_config'), exist_ok=True)
        for i,j in itertools.product(range(5), range(5)):
            conf['core_modules']['modules'][0]['models'][0]['weight_config'] = {
                'filename': os.path.join(results_dir, 'checkpoints', 'inner', f'forward-{i}.pt'),
                'freeze': True,
            }
            conf['core_modules']['modules'][1]['models'][0]['weight_config'] = {
                'filename': os.path.join(results_dir, 'checkpoints', 'inner', f'backward-{j}.pt'),
                'freeze': True,
            }
            with open(os.path.join(results_dir, 'model_config', f'h_{i}_{j}.yaml'), 'w') as f:
                yaml.dump(conf, f)

        # Train horizontal HRL model (with pre-trained inner model)
        direction = 'both'
        task_args = [
            [a for a in [
                '--env-config',
                    os.path.join(args.env_config_dir, f'train/{direction}/halfcheetah.yaml'),
                '--model-config',
                    os.path.join(results_dir, 'model_config', f'h_{i}_{j}.yaml'),
                '--optimizer', 'Adam',
                '--model-checkpoint',
                    os.path.join(results_dir, 'checkpoints', 'h_pt', '{RUN_ID}.pt'),
                '--checkpoint-interval', '1_000_000',
                '--run-id', f'{i}_{j}-{step}',
                '--wandb-id', f'{args.exp_id}_h_pt_{{RUN_ID}}',
                '--max-steps-total',
                    ('30000' if args.debug else str(args.max_steps_outer)),
                '--cuda',
                '--wandb' if args.wandb else None,
            ] if a is not None
        ] for step,(i,j) in enumerate(itertools.product(range(5), range(5)))]
        dependency = ':'.join(['afterany', *train_inner_job_ids]) if len(train_inner_job_ids) > 0 else None
        job_ids = run_train(
                task_args,
                job_name='train_h_pt',
                slurm=args.slurm,
                dependency=dependency,
                max_steps_per_job=5,
                duration=math.ceil(args.max_steps_outer / 15_000_000) + 2, # Run approximately 17M steps per day. Assume 15M and add 2 days as buffer.
        )
        train_job_ids.extend(job_ids)

    if 'train_v_pt' in args.actions:
        # Create model config for next step
        with open(os.path.join(args.model_config_dir, 'vertical.yaml'), 'r') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        os.makedirs(os.path.join(results_dir, 'model_config'), exist_ok=True)
        for i,j in itertools.product(range(5), range(5)):
            conf['core_modules']['modules'][0]['models'][0]['weight_config'] = {
                'filename': os.path.join(results_dir, 'checkpoints', 'inner', f'forward-{i}.pt'),
                'freeze': True,
            }
            conf['core_modules']['modules'][1]['models'][0]['weight_config'] = {
                'filename': os.path.join(results_dir, 'checkpoints', 'inner', f'backward-{j}.pt'),
                'freeze': True,
            }
            with open(os.path.join(results_dir, 'model_config', f'v_{i}_{j}.yaml'), 'w') as f:
                yaml.dump(conf, f)

        # Train vertical HRL model (with pre-trained inner model)
        direction = 'both'
        task_args = [
            [a for a in [
                '--env-config',
                    os.path.join(args.env_config_dir, f'train/{direction}/halfcheetah.yaml'),
                '--model-config',
                    os.path.join(results_dir, 'model_config', f'v_{i}_{j}.yaml'),
                '--optimizer', 'Adam',
                '--model-checkpoint',
                    os.path.join(results_dir, 'checkpoints', 'v_pt', '{RUN_ID}.pt'),
                '--checkpoint-interval', '1_000_000',
                '--run-id', f'{i}_{j}-{step}',
                '--wandb-id', f'{args.exp_id}_v_pt_{{RUN_ID}}',
                '--max-steps-total',
                    ('30000' if args.debug else str(args.max_steps_outer)),
                '--cuda',
                '--wandb' if args.wandb else None,
            ] if a is not None
        ] for step,(i,j) in enumerate(itertools.product(range(5), range(5)))]
        dependency = ':'.join(['afterany', *train_inner_job_ids]) if len(train_inner_job_ids) > 0 else None
        job_ids = run_train(
                task_args,
                job_name='train_v_pt',
                slurm=args.slurm,
                dependency=dependency,
                max_steps_per_job=5,
                duration=math.ceil(args.max_steps_outer / 15_000_000) + 2, # Run approximately 17M steps per day. Assume 15M and add 2 days as buffer.
        )
        train_job_ids.extend(job_ids)

    # TODO: Evaluation?

    print(f'Launched {len(train_job_ids)} jobs (plotting not counted)')
    if len(train_job_ids) > 0:
        print(f'Training job IDs: {train_job_ids}')


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('actions', type=str, nargs='*',
                        #choices=['train', 'eval_random', 'eval_train', 'eval_test', 'plot'],
                        default=['train_inner', 'train_h_tr', 'train_v_tr', 'train_h_pt', 'train_v_pt', 'train_lstm'],
                        help='')

    parser.add_argument('--exp-id', type=str, required=True,
                        help='Identifier for the current experiment set.')
    parser.add_argument('--max-steps-inner', type=int, default=50_000_000,
                        help='Maximum number of steps to train for (applies to inner models).')
    parser.add_argument('--max-steps-outer', type=int, default=50_000_000,
                        help='Maximum number of steps to train for (applies to models that contain the inner models and models that are directly compared to the former).')
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
