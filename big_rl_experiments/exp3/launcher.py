import argparse
from collections import defaultdict
import datetime
import glob
import os
import logging
import subprocess
import sys
import yaml

import numpy as np
import torch
from simple_slurm import Slurm

from big_rl_experiments.exp3.__main__ import main


logger = logging.getLogger(__name__)

EXP_NAME = 'exp3'
SHELL = '/bin/bash' # bash is needed because `trap` behaves differently in sh and the `module` command isn't available in sh.
GRES = 'gpu:a100l.2g.20gb:1'
TRAINING_STEPS = 50_000_000


def make_dependency_string(job_ids: list[int], dependency_type: str = 'afterany') -> str | None:
    if len(job_ids) == 0:
        return None
    return f'{dependency_type}:{":".join([str(x) for x in job_ids])}'


def relaunch(actions: list[str], job_name: str | None = None, slurm: bool = False, subproc: bool = False, dependency=None) -> list[int]:
    """ Call this script again with a different set of tasks. Useful in scenarios where the arguments for the task depends the results of another task. """
    script = f'big_rl_experiments/{EXP_NAME}/launcher.py'
    args = actions + get_args(with_tasks=False)

    if job_name is None:
        job_name = f'launch[{",".join(actions)}]'

    if slurm:
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=1,
            mem='1G',
            output='/network/scratch/h/huanghow/slurm/%A.out',

            time='1:00:00',

            **slurm_kwargs,
        )
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')

        cmd = f'python {script} {" ".join(args)}'
        print('-'*80)
        print(s.script(shell=SHELL))
        print(cmd)
        print('-'*80)
        job_id = s.sbatch(cmd, shell=SHELL)
        return [job_id]
    elif subproc:
        for a in args:
            cmd = f'python3 {script} {" ".join(args)}'

            print(cmd)

            p = subprocess.Popen(cmd, shell=True) # if shell=True, then the subprocesses will have the same environment variables as this process. Needed to pass the CUDA_VISIBLE_DEVICES variable.
            p.wait()
        return []
    else:
        launch(init_arg_parser().parse_args(args))
        return []


def run_train(args: list[list[str]], job_name='train', slurm: bool = False, subproc: bool = False, cpus_per_task=2, mem_per_task=3, gres: list | str | None = [GRES], dependency=None, max_steps_per_job: int = 5, duration: int = 5, time_per_job=datetime.timedelta(days=1, hours=0, minutes=0, seconds=0)) -> list[int]:
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

    if isinstance(gres, str):
        gres = [gres]
    elif gres == []:
        gres = None

    if slurm:
        if len(args) > max_steps_per_job:
            job_ids = []
            for i in range(0, len(args), max_steps_per_job):
                jid = run_train(
                        args[i:i+max_steps_per_job],
                        job_name=job_name,
                        slurm=slurm,
                        subproc=subproc,
                        cpus_per_task=cpus_per_task,
                        mem_per_task=mem_per_task,
                        gres=gres,
                        dependency=dependency,
                        max_steps_per_job=max_steps_per_job,
                        duration=duration,
                        time_per_job=time_per_job,
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
            cpus_per_task=cpus_per_task*len(args),
            mem=f'{mem_per_task*len(args)}G',
            gres=gres,
            output='/network/scratch/h/huanghow/slurm/%A.out',

            # Run for five days
            array=f'1-{duration}%1',
            time=time_per_job,

            #partition='main',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        s.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        if len(args) > 1:
            srun_args = ' '.join([
                    '--overlap',
                    f'--gres=gpu',
                    '--output', '/network/scratch/h/huanghow/slurm/%A_%s_%a.out',
            ])
            for a in args:
                a = [a_ for a_ in a if a_ is not None]
                cmd = f'srun {srun_args} python {script} {action} {" ".join(a)}'
                s.add_cmd(cmd + ' &')

            print('-'*80)
            print(s.script(shell=SHELL))
            print('wait')
            print('-'*80)

            job_id = s.sbatch('wait', shell=SHELL)
        else:
            a = [a_ for a_ in args[0] if a_ is not None]
            cmd = f'python {script} {action} {" ".join(a)}'
            print('-'*80)
            print(s.script(shell=SHELL))
            print(cmd)
            print('-'*80)
            job_id = s.sbatch(cmd, shell=SHELL)
        return [job_id]
    elif subproc:
        for a in args:
            cmd = f'python3 {script} {action} {" ".join([a_ for a_ in a if a_ is not None])}'

            print(cmd)

            p = subprocess.Popen(cmd, shell=True) # if shell=True, then the subprocesses will have the same environment variables as this process. Needed to pass the CUDA_VISIBLE_DEVICES variable.
            p.wait()
        return []
    else:
        for i,a in enumerate(args):
            os.environ.update({'SLURM_STEP_ID': str(i)})
            main([action, *[a_ for a_ in a if a_ is not None]])
        return []


def run_eval(args: list[list[str]], job_name='eval', slurm: bool = False, subproc: bool = False, mem_per_task=2, dependency=None) -> list[int]:
    """
    Args:
        args: List of lists of arguments to pass to the training script. Each sublist is a set of arguments to pass to a single training job.
        slurm: If True, submit the jobs to slurm. If False, run the jobs locally.
        subproc: If True, run the jobs locally as subprocesses. If False, run the jobs in the current process.
        mem_per_task: Memory per task in GB. Used only if `slurm` is True.
        dependency: Job ID to depend on. Used only if `slurm` is True.
        max_steps_per_job: Maximum number of steps to train for per job. Used only if `slurm` is True.
    """
    script = f'big_rl_experiments/{EXP_NAME}/__main__.py'
    action = 'eval'
    if slurm:
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=1,
            mem=f'{mem_per_task}G',
            #gres=[GRES],
            output='/network/scratch/h/huanghow/slurm/%A.out',

            time='12:00:00',

            partition='unkillable-cpu',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        job_ids = []
        for a in args:
            cmd = f'python {script} {action} {" ".join(a)}'

            print('-'*80)
            print(s.script(shell=SHELL))
            print(cmd)
            print('-'*80)

            job_id = s.sbatch(cmd, shell=SHELL)
            job_ids.append(job_id)
        return job_ids
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


def run_probe_module_beliefs(args: list[list[str]], job_name='analysis', slurm: bool = False, subproc: bool = False, mem_per_task=2, dependency=None) -> list[int]:
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
    action = 'analysis'
    if slurm:
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=1,
            mem=f'{mem_per_task}G',
            output='/network/scratch/h/huanghow/slurm/%A.out',

            # Ask for 1h (Should take about 5min)
            time='1:00:00',

            #partition='main',
            partition='unkillable-cpu',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        job_ids = []
        for a in args:
            cmd = f'srun python {script} {action} {" ".join(a)}'

            print('-'*80)
            print(s.script(shell=SHELL))
            print(cmd)
            print('-'*80)

            job_id = s.sbatch(cmd, shell=SHELL)
            job_ids.append(job_id)
        return job_ids
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


def run_plot_module_beliefs(args: list[list[str]], job_name='plot_module_beliefs', slurm: bool = False, subproc: bool = False, mem_per_task=2, dependency=None) -> list[int]:
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
    action = 'plot_module_beliefs'
    if slurm:
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=1,
            mem=f'{mem_per_task}G',
            output='/network/scratch/h/huanghow/slurm/%A.out',

            # Ask for 1h (Should take about 5min)
            time='1:00:00',

            #partition='main',
            partition='unkillable-cpu',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        job_ids = []
        for a in args:
            cmd = f'srun python {script} {action} {" ".join(a)}'

            print('-'*80)
            print(s.script(shell=SHELL))
            print(cmd)
            print('-'*80)

            job_id = s.sbatch(cmd, shell=SHELL)
            job_ids.append(job_id)
        return job_ids
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


def run_setup_ball_pt(args: list[str], job_name='setup_ball_pt', slurm: bool = False, subproc: bool = False, dependency=None) -> list[int]:
    ...


def run_plot_ball_learning_curve(args: list[str], job_name='plot_ball_learning_curve', slurm: bool = False, subproc: bool = False, dependency=None) -> list[int]:
    script = f'big_rl_experiments/{EXP_NAME}/__main__.py'
    action = 'plot_ball_learning_curve'
    if slurm:
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=1,
            mem=f'1G',
            output='/network/scratch/h/huanghow/slurm/%A.out',

            # Ask for 1h
            time='1:00:00',

            #partition='main',
            partition='unkillable-cpu',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        #slurm.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        
        cmd = f'srun python {script} {action} {" ".join(args)}'

        print('-'*80)
        print(s.script(shell=SHELL))
        print(cmd)
        print('-'*80)

        job_id = s.sbatch(cmd, shell=SHELL)
        return [job_id]
    elif subproc:
        cmd = f'python3 {script} {action} {" ".join(args)}'

        print(cmd)

        p = subprocess.Popen(cmd, shell=True) # if shell=True, then the subprocesses will have the same environment variables as this process. Needed to pass the CUDA_VISIBLE_DEVICES variable.
        p.wait()
        return []
    else:
        main([action, *args])
        return []


def run_archive(args: list[str], job_name='archive', slurm: bool = False, subproc: bool = False, dependency=None) -> list[int]:
    script = f'big_rl_experiments/{EXP_NAME}/__main__.py'
    action = 'archive'
    if slurm:
        slurm_kwargs = {}
        if dependency is not None:
            slurm_kwargs['dependency'] = dependency
        s = Slurm(
            job_name=job_name,
            cpus_per_task=1,
            mem=f'1G',
            output='/network/scratch/h/huanghow/slurm/%A.out',

            # Ask for 1h
            time='1:00:00',

            #partition='main',
            partition='unkillable-cpu',
            signal='USR1@120', # Send a signal to the job 120 seconds before it is killed

            **slurm_kwargs,
        )
        s.add_cmd('module load libffi') # Fixes the "ImportError: libffi.so.6: cannot open shared object file: No such file or directory" error
        s.add_cmd('module load python/3.10')
        s.add_cmd('source ENV2204/bin/activate')
        s.add_cmd('export PYTHONUNBUFFERED=1')
        #ps://stackoverflow.com/questions/76348342/how-to-use-trap-in-my-sbatch-bash-job-script-in-compute-canada
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGUSR1") # Handles time limit
        s.add_cmd("trap 'echo SIGUSR1 1>&2' SIGTERM") # Handles preemption (I think this is needed if PreemptParameters isn't set with send_user_signal enabled. Check if it's set in /etc/slurm/slurm.conf)
        
        cmd = f'srun python {script} {action} {" ".join(args)}'

        print('-'*80)
        print(s.script(shell=SHELL))
        print(cmd)
        print('-'*80)

        job_id = s.sbatch(cmd, shell=SHELL)
        return [job_id]
    elif subproc:
        cmd = f'python3 {script} {action} {" ".join(args)}'

        print(cmd)

        p = subprocess.Popen(cmd, shell=True) # if shell=True, then the subprocesses will have the same environment variables as this process. Needed to pass the CUDA_VISIBLE_DEVICES variable.
        p.wait()
        return []
    else:
        main([action, *args])
        return []


def get_args(with_tasks: bool = True):
    args = sys.argv[1:]
    if not with_tasks:
        while not args[0].startswith('--'):
            args = args[1:]
    return args


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

    def make_launcher_task(task_name, dependency = None):
        if f'{task_name}--launcher' in args.actions:
            job_ids = relaunch(
                actions = [task_name],
                dependency = dependency,
                slurm = args.slurm,
                subproc = args.subproc,
            )
            job_ids_by_task[task_name].extend(job_ids)

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
                duration=7,
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
                duration=7,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    # Multitask training (pre-trained, no freezing)
    task_name = 'train_multi_task_pt'
    if task_name in args.actions:
        starting_models_dir = os.path.join(results_dir, 'checkpoints', 'train_single_task')
        #starting_models = [os.path.join(starting_models_dir, f'{i}.pt') for i in range(5)]
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
                dependency=make_dependency_string(job_ids_by_task['train_single_task']),
                max_steps_per_job=5,
                duration=7,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    # Multitask training (pre-trained, frozen core modules)
    task_name = 'train_multi_task_pt_frozen'
    if task_name in args.actions:
        starting_models_dir = os.path.join(results_dir, 'checkpoints', 'train_single_task')
        #starting_models = [os.path.join(starting_models_dir, f'{i}.pt') for i in range(5)]
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
                dependency=make_dependency_string(job_ids_by_task['train_single_task']),
                max_steps_per_job=5,
                duration=5,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    # Evaluation (eval each model on test task so we can compare their performance to the mutual info between hiden state and task)
    make_launcher_task(
            'eval',
            dependency = make_dependency_string(sum([job_ids_by_task[k] for k in ['train_multi_task_pt', 'train_multi_task_pt_frozen', 'train_multi_task_tr', 'train_single_task']], [])),
    )

    task_name = 'eval'
    if task_name in args.actions:
        checkpoint_subdirs = ['train_multi_task_pt', 'train_multi_task_pt_frozen', 'train_multi_task_tr', 'train_single_task']
        checkpoint_dir = os.path.join(results_dir, 'checkpoints')
        task_args=[
            [
                '--env-config',
                    os.path.join(args.env_config_dir, 'test/halfcheetah_single.yaml'),
                '--model-config',
                    os.path.join(args.model_config_dir, 'model.yaml'),
                '--model',
                    os.path.join(checkpoint_dir, subdir, checkpoint),
                '--no-video',
                '--num-episodes', '10',
                '--results', os.path.join(results_dir, 'eval', subdir, checkpoint),
                '--cache-results',
            ]
            for subdir in checkpoint_subdirs
            for checkpoint in os.listdir(os.path.join(checkpoint_dir, subdir))
        ]
        job_ids = run_eval(
                task_args,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                dependency=make_dependency_string(sum([job_ids_by_task[k] for k in checkpoint_subdirs], [])),
        )
        job_ids_by_task[task_name].extend(job_ids)

    task_name = 'probe_module_beliefs'
    if task_name in args.actions:
        checkpoint_subdirs = ['train_multi_task_pt', 'train_multi_task_pt_frozen', 'train_multi_task_tr', 'train_single_task']
        task_args = [[
            '--model-config-dir', args.model_config_dir,
            '--checkpoint',
                os.path.join(results_dir, 'checkpoints', subdir, f'{i}.pt'),
            '--results-dir', os.path.join(results_dir, 'analysis', subdir, str(i)),
            '--num-episodes', '50',
            '--num-epochs', '10_000',
            '--cache-dataset',
        ] for subdir in checkpoint_subdirs for i in range(5)]
        job_id = run_probe_module_beliefs(
                task_args,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                mem_per_task=2,
                dependency=make_dependency_string(sum([job_ids_by_task[k] for k in checkpoint_subdirs], [])),
        )
        job_ids_by_task[task_name].extend(job_id)

    task_name = 'plot_module_beliefs'
    if task_name in args.actions:
        checkpoint_subdirs = ['train_multi_task_pt', 'train_multi_task_pt_frozen', 'train_multi_task_tr', 'train_single_task']
        task_args = [[
            '--results-dir', results_dir,
            '--output-dir',
                os.path.join(results_dir, 'plots'),
        ]]
        job_id = run_plot_module_beliefs(
                task_args,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                mem_per_task=2,
                dependency=make_dependency_string(job_ids_by_task['probe_module_beliefs']),
        )
        job_ids_by_task[task_name].extend(job_id)

    task_name = 'results'
    if task_name in args.actions:
        # Aggregate results
        eval_results = {}
        for filename in glob.iglob(os.path.join(results_dir, 'eval', '**', '*.pt'), recursive=True):
            result = torch.load(filename)
            key = '/'.join(filename.split('/')[-2:])[:-3]
            eval_results[key] = {
                'fw': np.mean([r['episode_reward'] for r in result['fw']]),
                'bw': np.mean([r['episode_reward'] for r in result['bw']]),
            }

        anal_results = {}
        for filename in glob.iglob(os.path.join(results_dir, 'analysis', '**', 'results.pt'), recursive=True):
            result = torch.load(filename)
            key = '/'.join(filename.split('/')[-3:-1])
            anal_results[key] = result
        
        # Look at the results below:
        # {k: [f'{x:.3}' for x in v] for k,v in anal_results.items()}
        # eval_results
        breakpoint()

    task_name = 'train_ball_tr'
    if task_name in args.actions:
        task_args = [[
            '--env-config',
                os.path.join(args.env_config_dir, 'train/ball.yaml'),
            '--model-config',
                os.path.join(args.model_config_dir, 'model-ball.yaml'),
            '--optimizer', 'Adam',
            '--model-checkpoint',
                os.path.join(results_dir, 'checkpoints', task_name, '{RUN_ID}.pt'),
            '--checkpoint-interval', '1_000_000',
            '--max-steps-total', '10_000_000',
            '--eval-interval', '100_000',
            '--eval-episodes', '10',
            '--eval-results-dir',
                os.path.join(results_dir, 'learning_curve', task_name, '{RUN_ID}'),
            '--run-id', f'{i}',
            '--wandb-id', f'{args.exp_id}__{task_name}__{{RUN_ID}}',
            '--cuda',
            '--wandb' if args.wandb else None,
            '--wandb-project', f'big_rl_{EXP_NAME}',
            '--wandb-group', task_name,
        ] for i in range(5)]
        task_args = [a for a in task_args if a is not None]
        job_ids = run_train(
                task_args,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                cpus_per_task=1,
                mem_per_task=3,
                gres=None,
                max_steps_per_job=1,
                duration=2,
                time_per_job=time_per_job,
        )
        job_ids_by_task[task_name].extend(job_ids)

    make_launcher_task(
            'train_ball_pt',
            dependency=make_dependency_string(job_ids_by_task['train_multi_task_pt_frozen'])
    )

    task_name = 'train_ball_pt'
    if task_name in args.actions:
        # Generate model config files
        model_config_template_path = os.path.join(args.model_config_dir, 'model-ball-pt.yaml')
        with open(model_config_template_path, 'r') as f:
            model_config_template = yaml.safe_load(f)
        #checkpoint_subdirs = ['train_multi_task_pt', 'train_multi_task_pt_frozen', 'train_multi_task_tr', 'train_single_task']
        checkpoint_subdirs = ['train_multi_task_pt_frozen']
        checkpoint_dir = os.path.join(results_dir, 'checkpoints')
        task_args = []
        for subdir in checkpoint_subdirs:
            for checkpoint in os.listdir(os.path.join(checkpoint_dir, subdir)):
                checkpoint_filename = os.path.join(checkpoint_dir, subdir, checkpoint)
                checkpoint_num = int(checkpoint.split('.')[0])
                model_config_template['input_modules']['reward']['weight_config']['filename'] = checkpoint_filename
                model_config_template['core_modules']['weight_config']['filename'] = checkpoint_filename
                for i in range(3):
                    model_config_template['core_modules']['weight_config']['key_prefix'] = f'core_modules.{i}'
                    model_config_filename = os.path.join(results_dir, 'eval_ball', 'model_configs', subdir, f'{checkpoint_num}_{i}.yaml')
                    os.makedirs(os.path.dirname(model_config_filename), exist_ok=True)
                    with open(model_config_filename, 'w') as f:
                        yaml.dump(model_config_template, f)
                    new_checkpoint_filename = os.path.join(results_dir, 'checkpoints', task_name, subdir, f'{checkpoint_num}_{i}.pt')
                    print(model_config_filename)
                    task_args.append([
                        '--env-config',
                            os.path.join(args.env_config_dir, 'train/ball.yaml'),
                        '--model-config',
                            model_config_filename,
                        '--optimizer', 'Adam',
                        '--model-checkpoint',
                            new_checkpoint_filename,
                        '--checkpoint-interval', '1_000_000',
                        '--max-steps-total', '10_000_000',
                        '--eval-interval', '100_000',
                        '--eval-episodes', '10',
                        '--eval-results-dir',
                            os.path.join(results_dir, 'learning_curve', task_name, '{RUN_ID}'),
                        '--run-id', f'{subdir}{checkpoint_num}_m{i}',
                        '--wandb-id', f'{args.exp_id}__{task_name}__{{RUN_ID}}',
                        '--cuda',
                        '--wandb' if args.wandb else None,
                        '--wandb-project', f'big_rl_{EXP_NAME}',
                        '--wandb-group', task_name,
                    ])
        # Train with these models
        job_ids = run_train(
                task_args,
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                cpus_per_task=1,
                mem_per_task=3,
                gres=None,
                max_steps_per_job=1,
                duration=2,
                time_per_job=time_per_job,
                dependency=make_dependency_string(sum([job_ids_by_task[k] for k in checkpoint_subdirs], [])),
        )
        job_ids_by_task[task_name].extend(job_ids)

    make_launcher_task(
            'train_ball_pt',
            dependency=make_dependency_string(job_ids_by_task['train_multi_task_pt_frozen'])
    )

    task_name = 'plot_ball_learning_curve'
    if task_name in args.actions:
        task_args = [
            '--plot-dir', os.path.join(results_dir, 'learning_curve'),
            '--directories-tr', *[
                os.path.join(results_dir, 'learning_curve', 'train_ball_tr', str(i))
                for i in range(5)
            ],
            '--directories-pt-0', *[
                os.path.join(results_dir, 'learning_curve', 'train_ball_pt', f'train_multi_task_pt_frozen{i}_m0')
                for i in range(5)
            ],
            '--directories-pt-1', *[
                os.path.join(results_dir, 'learning_curve', 'train_ball_pt', f'train_multi_task_pt_frozen{i}_m1')
                for i in range(5)
            ],
            '--directories-pt-2', *[
                os.path.join(results_dir, 'learning_curve', 'train_ball_pt', f'train_multi_task_pt_frozen{i}_m2')
                for i in range(5)
            ],
        ]
        run_plot_ball_learning_curve(
                task_args, job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                dependency=make_dependency_string(job_ids_by_task['train_ball_pt'] + job_ids_by_task['train_ball_tr']),
        )

    make_launcher_task(
            'plot_ball_learning_curve',
            dependency=make_dependency_string(job_ids_by_task['train_ball_pt'] + job_ids_by_task['train_ball_tr']),
    )

    task_name = 'archive'
    if task_name in args.actions:
        if 'ARCHIVE' not in os.environ:
            print('ARCHIVE environment variable not set. Skipping archiving.')
        archive_dir = os.path.join(os.environ['ARCHIVE'], 'results', 'big_rl_experiments', EXP_NAME)
        archive_file = os.path.join(archive_dir, f'{args.exp_id}.tar')
        run_archive(
                [results_dir, archive_file],
                job_name=task_name,
                slurm=args.slurm,
                subproc=args.subproc,
                dependency=make_dependency_string(sum(job_ids_by_task.values(), [])),
        )

    print(f'Launched {sum(len(v) for v in job_ids_by_task.values())} jobs')
    for k,v in job_ids_by_task.items():
        print(f'{k}: {v}')


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('actions', type=str, nargs='*',
                        default=['train_single_task', 'train_multi_task_tr', 'train_multi_task_pt', 'train_multi_task_pt_frozen', 'eval--launcher', 'analyse', 'train_ball_tr', 'train_ball_pt--launcher', 'plot_ball_learning_curve--launcher'],
                        help='')

    parser.add_argument('--exp-id', type=str, required=True,
                        help='Identifier for the current experiment set.')
    parser.add_argument('--max-steps-single-task',
                        type=int, default=50_000_000,
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
