"""
A script to evaluate the plasticity loss in the baseline setup.
This is done by training the model on multiple tasks in sequence, and observing the peak performance each time a task is revisited.

The script is a little complicated because it was copied from the main training script with some things removed, but not reorganized.
"""

import itertools
from typing import Optional, Generator, Dict, List, Any, Callable
import time

import gymnasium
import gymnasium.spaces
import gymnasium.vector
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn
import torch.nn.utils
from tensordict import TensorDict
import numpy as np
import wandb
from minigrid.core.constants import COLOR_NAMES

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss

from big_rl.minigrid.envs import make_env
from big_rl.utils import merge_space
from big_rl.minigrid.common import init_model, env_config_presets


def enum_minibatches(batch_size, minibatch_size, num_minibatches, replace=False):
    indices = np.arange(batch_size)
    if replace:
        for _ in range(0,batch_size,minibatch_size):
            np.random.shuffle(indices)
            yield indices[:minibatch_size]
    else:
        indices = np.arange(batch_size)
        n = batch_size//minibatch_size
        for i in range(num_minibatches):
            j = i % n
            if j == 0:
                np.random.shuffle(indices)
            yield indices[j*minibatch_size:(j+1)*minibatch_size]


def compute_ppo_losses(
        observation_space : gymnasium.spaces.Dict,
        action_space : gymnasium.Space,
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        preprocess_input_fn: Optional[Callable],
        discount : float,
        gae_lambda : float,
        norm_adv : bool,
        clip_vf_loss : Optional[float],
        entropy_loss_coeff : float,
        vf_loss_coeff : float,
        target_kl : Optional[float],
        minibatch_size : int,
        num_minibatches : int) -> Generator[Dict[str,torch.Tensor],None,None]:
    """
    Compute the losses for PPO.
    """
    device = next(model.parameters()).device

    obs = history.obs
    action = history.action
    reward = history.reward
    terminal = history.terminal

    n = len(history.obs_history)
    num_training_envs = len(history.reward_history[0])

    if preprocess_input_fn is None:
        preprocess_input_fn = lambda x: x

    with torch.no_grad():
        obs_tensordict = TensorDict(preprocess_input_fn(obs), batch_size=(n,), device=device)
        net_output = default_collate([model(o) for o in obs_tensordict])
        state_values_old = net_output['value'].squeeze(2)
        action_dist = torch.distributions.Categorical(logits=net_output['action'][:n-1])
        log_action_probs_old = action_dist.log_prob(action)

        # Advantage
        advantages = generalized_advantage_estimate(
                state_values = state_values_old[:n-1,:],
                next_state_values = state_values_old[1:,:],
                rewards = reward[1:,:],
                terminals = terminal[1:,:],
                discount = discount,
                gae_lambda = gae_lambda,
        )
        returns = advantages + state_values_old[:n-1,:]

    # Flatten everything
    flat_obs = preprocess_input_fn({
            k:v[:n-1].reshape(-1,*observation_space[k].shape)
            for k,v in obs.items()
    })
    flat_obs = TensorDict(preprocess_input_fn({
            k:v[:n-1].reshape(-1,*observation_space[k].shape)
            for k,v in obs.items()
    }), batch_size=((n-1)*num_training_envs,), device=device)
    flat_action = action[:n-1].reshape(-1, *action_space.shape)
    flat_terminals = terminal[:n-1].reshape(-1)
    flat_returns = returns[:n-1].reshape(-1)
    flat_advantages = advantages[:n-1].reshape(-1)
    flat_log_action_probs_old = log_action_probs_old[:n-1].reshape(-1)
    flat_state_values_old = state_values_old[:n-1].reshape(-1)

    minibatches = enum_minibatches(
            batch_size=(n-1) * num_training_envs,
            minibatch_size=minibatch_size,
            num_minibatches=num_minibatches,
            replace=False)

    for _,mb_inds in enumerate(minibatches):
        mb_obs = flat_obs[mb_inds]
        mb_action = flat_action[mb_inds]
        mb_returns = flat_returns[mb_inds]
        mb_advantages = flat_advantages[mb_inds]
        mb_terminals = flat_terminals[mb_inds]
        mb_log_action_probs_old = flat_log_action_probs_old[mb_inds]
        mb_state_values_old = flat_state_values_old[mb_inds]

        net_output = model(mb_obs)
        assert 'value' in net_output
        assert 'action' in net_output
        mb_state_values = net_output['value'].squeeze()
        action_dist = torch.distributions.Categorical(logits=net_output['action'])
        mb_log_action_probs = action_dist.log_prob(mb_action)
        mb_entropy = action_dist.entropy()

        with torch.no_grad():
            logratio = mb_log_action_probs - mb_log_action_probs_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()

        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss = clipped_advantage_policy_gradient_loss(
                log_action_probs = mb_log_action_probs,
                old_log_action_probs = mb_log_action_probs_old,
                advantages = mb_advantages,
                terminals = mb_terminals,
                epsilon=0.1
        ).mean()

        # Value loss
        if clip_vf_loss is not None:
            v_loss_unclipped = (mb_state_values - mb_returns) ** 2
            v_clipped = mb_state_values_old + torch.clamp(
                mb_state_values - mb_state_values_old,
                -clip_vf_loss,
                clip_vf_loss,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((mb_state_values - mb_returns) ** 2).mean()

        entropy_loss = mb_entropy.mean()
        loss = pg_loss - entropy_loss_coeff * entropy_loss + v_loss * vf_loss_coeff

        yield {
                'loss': loss,
                'loss_pi': pg_loss,
                'loss_vf': v_loss,
                'loss_entropy': -entropy_loss,
                'approx_kl': approx_kl,
                'state_value': mb_state_values,
                'entropy': mb_entropy,
        }

        if target_kl is not None:
            if approx_kl > target_kl:
                break


def train_single_env(
        step_counter: Dict[str,int],
        model: torch.nn.Module,
        env: gymnasium.vector.VectorEnv,
        env_label: str,
        env_index: int,
        *,
        obs_scale: Dict[str,float] = {},
        obs_ignore: List[str] = ['obs (mission)'],
        rollout_length: int = 128,
        reward_scale: float = 1.0,
        reward_clip: Optional[float] = 1.0,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_vf_loss: Optional[float] = None,
        entropy_loss_coeff: float = 0.01,
        vf_loss_coeff: float = 0.5,
        num_minibatches: int = 4,
        minibatch_size: int = 32,
        target_kl: Optional[float] = None,
        norm_adv: bool = True,
        ) -> Generator[Dict[str, Any], None, None]:
    """
    Train a model with PPO on an Atari game.

    Args:
        model: ...
        env: `gym.vector.VectorEnv`
    """
    num_envs = env.num_envs

    device = next(model.parameters()).device
    observation_space = env.single_observation_space
    action_space = env.single_action_space
    assert isinstance(observation_space, gymnasium.spaces.Dict)

    def preprocess_input(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.tensor(v, device=device)*obs_scale.get(k,1)
            for k,v in obs.items() if k not in obs_ignore
        }

    history = VecHistoryBuffer(
            num_envs = num_envs,
            max_len=rollout_length+1,
            device=device)
    log = {}

    obs, info = env.reset(seed=0)
    history.append_obs(
            {k:v for k,v in obs.items() if k not in obs_ignore},
    )
    episode_true_reward = np.zeros(num_envs) # Actual reward we want to optimize before any modifications (e.g. clipping)
    episode_reward = np.zeros(num_envs) # Reward presented to the learning algorithm
    episode_steps = np.zeros(num_envs)

    ##################################################
    # Start training
    for step in itertools.count():
        # Gather data
        state_values = [] # For logging purposes
        entropies = [] # For logging purposes
        for _ in range(rollout_length):
            step_counter[env_label] += num_envs
            step_counter['total'] += num_envs

            # Select action
            with torch.no_grad():
                model_output = model({
                    k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
                    for k,v in obs.items()
                    if k not in obs_ignore
                })

                action_probs = model_output['action'].softmax(1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().cpu().numpy()

                state_values.append(model_output['value'])
                entropies.append(action_dist.entropy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated | truncated

            history.append_action(action)
            episode_reward += reward
            episode_true_reward += info.get('reward', reward)
            episode_steps += 1

            reward *= reward_scale
            if reward_clip is not None:
                reward = np.clip(reward, -reward_clip, reward_clip)

            history.append_obs(
                    {k:v for k,v in obs.items() if k not in obs_ignore}, reward, done,
            )

            if done.any():
                print(f'Episode finished ({step * num_envs * rollout_length:,} -- {step_counter["total"]:,})')
                if wandb.run is not None:
                    data = {
                            f'reward': episode_reward[done].mean().item(),
                            f'reward/{env_label}': episode_reward[done].mean().item(),
                            f'reward/by_index/{env_index}': episode_reward[done].mean().item(),
                            f'true_reward': episode_true_reward[done].mean().item(),
                            f'true_reward/{env_label}': episode_true_reward[done].mean().item(),
                            f'true_reward/by_index/{env_index}': episode_true_reward[done].mean().item(),
                            f'episode_length/{env_label}': episode_steps[done].mean().item(),
                            f'episode_length/by_index/{env_index}': episode_steps[done].mean().item(),
                            f'env_step/{env_label}': step_counter[env_label],
                            f'env_step/by_index/{env_index}': step_counter[env_label],
                            'step_total': step_counter['total'],
                    }
                    wandb.log(data, step = step_counter['total'])
                print(f'  reward: {episode_reward[done].mean():.2f}\t len: {episode_steps[done].mean()} \t env: {env_label} ({done.sum().item()})')
                # Reset episode stats
                episode_reward[done] = 0
                episode_true_reward[done] = 0
                episode_steps[done] = 0

        # Train
        losses = compute_ppo_losses(
                observation_space=observation_space,
                action_space=action_space,
                history = history,
                model = model,
                preprocess_input_fn = preprocess_input,
                discount = discount,
                gae_lambda = gae_lambda,
                norm_adv = norm_adv,
                clip_vf_loss = clip_vf_loss,
                vf_loss_coeff = vf_loss_coeff,
                entropy_loss_coeff = entropy_loss_coeff,
                target_kl = target_kl,
                minibatch_size = minibatch_size,
                num_minibatches = num_minibatches,
        )
        x = None
        for x in losses:
            log['state_value'] = torch.stack(state_values)
            log['entropy'] = torch.stack(entropies)

            yield {
                'log': log,
                **x
            }

            log = {}

        # Clear data
        history.clear()


def train(
        model: torch.nn.Module,
        env: gymnasium.vector.VectorEnv,
        env_label: str,
        env_index: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # XXX: Private class. This might break in the future.
        *,
        rollout_length: int = 128,
        obs_scale: Dict[str,float] = {},
        reward_scale: float = 1.0,
        reward_clip: Optional[float] = 1.0,
        max_grad_norm: float = 0.5,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_vf_loss: Optional[float] = None,
        entropy_loss_coeff: float = 0.01,
        vf_loss_coeff: float = 0.5,
        num_minibatches: int = 4,
        minibatch_size: int = 32,
        target_kl: Optional[float] = None,
        norm_adv: bool = True,
        step_counter: Dict[str,int] = {},
        ):
    trainer = train_single_env(
        step_counter = step_counter,
        model = model,
        env = env,
        env_label = env_label,
        env_index = env_index,
        obs_scale = obs_scale,
        rollout_length = rollout_length,
        reward_scale = reward_scale,
        reward_clip = reward_clip,
        discount = discount,
        gae_lambda = gae_lambda,
        clip_vf_loss = clip_vf_loss,
        entropy_loss_coeff = entropy_loss_coeff,
        vf_loss_coeff = vf_loss_coeff,
        minibatch_size = minibatch_size,
        num_minibatches = num_minibatches,
        target_kl = target_kl,
        norm_adv = norm_adv,
    )

    for _, loss in enumerate(trainer):
        if 'loss' not in loss:
            yield {
                'step': step_counter['total'],
                'episode_rewards': loss['episode_rewards'],
            }
            return

        optimizer.zero_grad()
        loss['loss'].backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # type: ignore
        for p in model.parameters(): # XXX: Debugging code
            if p._grad is None:
                continue
            if p._grad.isnan().any():
                print('NaNs in gradients!')
                breakpoint()
        optimizer.step()

        if wandb.run is not None:
            data = {
                f'loss/pi/{env_label}': loss['loss_pi'].item(),
                f'loss/v/{env_label}': loss['loss_vf'].item(),
                f'loss/entropy/{env_label}': loss['loss_entropy'].item(),
                f'loss/total/{env_label}': loss['loss'].item(),
                f'approx_kl/{env_label}': loss['approx_kl'].item(),
                f'state_value/{env_label}': loss['log']['state_value'].mean().item(),
                f'entropy/{env_label}': loss['log']['entropy'].mean().item(),
                #last_approx_kl=approx_kl.item(),
                #'learning_rate': lr_scheduler.get_lr()[0],
                f'env_step/{env_label}': step_counter[env_label],
                f'env_step/by_index/{env_index}': step_counter[env_label],
                'step_total': step_counter['total'],
            }
            wandb.log(data, step = step_counter['total'])

        yield {
            #'step': global_step_counter[0],
        }

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()


def finetune(
        env_config_names: List[str],
        env_labels: List[str],
        num_envs: int,
        checkpoint_filename: str,
        model_type: str,
        model_architecture: List[int],
        # Training parameters
        optimizer_type: str = 'adam',
        learning_rate: float = 1e-5,
        max_steps: int = 0,
        rollout_length: int = 128,
        reward_clip: Optional[float] = 1.0,
        reward_scale: float = 1.0,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        norm_adv: bool = True,
        clip_vf_loss: Optional[float] = 0.1,
        vf_loss_coeff: float = 0.5,
        entropy_loss_coeff: float = 0.01,
        minibatch_size: int = 256,
        num_minibatches: int = 4,
        target_kl: Optional[float] = None,
        max_grad_norm: float = 0.5,
        cuda: bool = True,
        load_optimizer: bool = False,
        # Plasticity experiment config
        steps_per_env: int = 1_000,
        num_cycles: int = 10,
        cycle_task_stop_index: Optional[int] = None,
        num_tasks: Optional[int] = None,
        # Override configs
        num_trials: Optional[int] = None,
        room_size: Optional[int] = None,
    ):
    ENV_CONFIG_PRESETS = env_config_presets()
    envs = {}
    for config_name,label in zip(env_config_names,env_labels):
        env_config = ENV_CONFIG_PRESETS[config_name]
        if num_trials is not None:
            env_config['config']['num_trials'] = num_trials
        if room_size is not None:
            env_config['config']['min_room_size'] = room_size
            env_config['config']['max_room_size'] = room_size
        envs[label] = gymnasium.vector.SyncVectorEnv([lambda: make_env(**env_config) for _ in range(num_envs)])

    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize model
    model = init_model(
            observation_space = merge_space(*[env.single_observation_space for env in envs.values()]),
            action_space = next(iter(envs.values())).single_action_space, # Assume the same action space for all environments
            model_type = model_type,
            recurrence_type = None,
            architecture = model_architecture,
            device = device,
    )
    model.to(device)

    # Initialize optimizer
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_type}')

    # Initialize learning rate scheduler
    lr_scheduler = None # TODO

    # Load checkpoint
    checkpoint = None

    if checkpoint_filename is not None:
        checkpoint = torch.load(checkpoint_filename, map_location=device)
        print(f'Loading model from {checkpoint_filename}')
    else:
        print('No checkpoint specified. Using random weights.')

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and load_optimizer:
            breakpoint()
            optimizer.load_state_dict(checkpoint['optimizer'])
        #if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        #    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_step = checkpoint.get('step', 0)
        print(f'Loaded checkpoint from {checkpoint_filename} at step {start_step}')

    # Random order
    env_order = list(envs.keys())
    np.random.shuffle(env_order)
    if num_tasks is not None:
        env_order = env_order[:num_tasks]
    print(f'Environment order: {env_order}')

    step_counter = { k: 0 for k in envs.keys() }
    step_counter['total'] = 0

    # Initialize trainer
    start_time = time.time()
    for c in range(num_cycles):
        for task_idx, env_key in enumerate(env_order):
            print('-'*80)
            print(f'Environment: {env_key}')
            print('-'*80)

            trainer = train(
                    model = model,
                    env = envs[env_key],
                    env_label = env_key,
                    env_index = task_idx,
                    optimizer = optimizer,
                    lr_scheduler = lr_scheduler,
                    rollout_length = rollout_length,
                    obs_scale = {'obs (image)': 1.0/255.0},
                    reward_clip = reward_clip,
                    reward_scale = reward_scale,
                    discount = discount,
                    gae_lambda = gae_lambda,
                    norm_adv = norm_adv,
                    clip_vf_loss = clip_vf_loss,
                    vf_loss_coeff = vf_loss_coeff,
                    entropy_loss_coeff = entropy_loss_coeff,
                    target_kl = target_kl,
                    minibatch_size = minibatch_size,
                    num_minibatches = num_minibatches,
                    max_grad_norm = max_grad_norm,
                    step_counter = step_counter,
            )

            # Run training loop
            for i,_ in enumerate(trainer):
                if i >= steps_per_env:
                    # If the `cycle_task_stop_index` is specified, it means we want to stop cycling at that task index on the last cycle and keep training forever on that task.
                    if cycle_task_stop_index is not None and c >= num_cycles-1 and task_idx == cycle_task_stop_index:
                        pass
                    else:
                        break
                # Timing
                steps = step_counter['total']
                if steps > 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = steps / elapsed_time
                    if max_steps > 0:
                        remaining_time = int((max_steps - steps) / steps_per_sec)
                        remaining_hours = remaining_time // 3600
                        remaining_minutes = (remaining_time % 3600) // 60
                        remaining_seconds = (remaining_time % 3600) % 60
                        print(f"Step {steps:,}/{max_steps:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
                    else:
                        elapsed_time = int(elapsed_time)
                        elapsed_hours = elapsed_time // 3600
                        elapsed_minutes = (elapsed_time % 3600) // 60
                        elapsed_seconds = (elapsed_time % 3600) % 60
                        print(f"Step {steps:,} \t {int(steps_per_sec):,} SPS \t Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")

            print(f'Done. {step_counter[env_key]:,} steps on {env_key}.')

    return {
        #'episode_rewards': x['episode_rewards'],
    }


if __name__ == '__main__':
    import argparse

    from big_rl.minigrid.arguments import init_parser_trainer, init_parser_model

    parser = argparse.ArgumentParser()

    parser.add_argument('--run-id', type=str, default=None,
                        help='Identifier for the current experiment. This value is used for setting the "{RUN_ID}" variable. If not specified, then either a slurm job ID is used, or the current date and time, depending on what is available. Note that the time-based ID has a resolution of 1 second, so if two runs are started within less than a second of each other, they may end up with the same ID.')

    parser.add_argument('--envs', type=str,
            default=[
                f'fetch2-004-stop_100_trials-{obj_color}_{obj_type}-2'
                for obj_color, obj_type in itertools.product(COLOR_NAMES, ['ball', 'key'])
            ],
            nargs='*',
            help='Environments to train on. Each environment will run for a fixed number of steps before cycling to the next in a random order.')
    parser.add_argument('--num-envs', type=int, default=16,
            help='Number of environments to train on.')
    parser.add_argument('--steps-per-env', type=int, default=1_000,
            help='Number of training steps before cycling to the next environment.')
    parser.add_argument('--env-labels', type=str,
            default=[f'{obj_color}_{obj_type}' for obj_color, obj_type in itertools.product(COLOR_NAMES, ['ball', 'key'])],
            nargs='*', help='')
    parser.add_argument('--num-cycles', type=int, default=10,
                        help='Number of times to train on each environment')
    parser.add_argument('--cycle-task-stop-index', type=int, default=None,
                        help='The index of the task on which the training loop will stop on. The training loop cycles through each environment `--num-cycles` times. If `--cycle-task-stop-index` is specified, then on the last cycle once this task is reached, the training loop will stay on this task and continue training until the process is killed.')
    parser.add_argument('--num-tasks', type=int, default=None,
                        help='Number of tasks to train on. If specified, this many tasks will be selected from the list of tasks provided in --envs.')

    init_parser_trainer(parser)
    init_parser_model(parser)

    parser.add_argument('--starting-model', type=str, default=None,
                        help='Path to a model checkpoint to start training from.')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true', help='Save results to W&B.')
    parser.add_argument('--wandb-id', type=str, default=None,
                        help='W&B run ID.')

    # Override configs
    parser.add_argument('--load-optimizer', action='store_true', help='Load optimizer state from checkpoint.')
    parser.add_argument('--num-trials', type=int, default=None)
    parser.add_argument('--room-size', type=int, default=None)

    # Parse arguments
    args = parser.parse_args()

    if len(args.envs) != len(args.env_labels):
        raise ValueError('Must specify the same number of envs and env labels')

    if args.wandb:
        if args.wandb_id is not None:
            wandb.init(project='big_rl-mtft-plasticity', id=args.wandb_id, resume='allow')
        else:
            wandb.init(project='big_rl-mtft-plasticity')
        wandb.config.update(args, allow_val_change=True)

    results = finetune(
            env_config_names = args.envs,
            env_labels = args.env_labels,
            num_envs = args.num_envs,
            checkpoint_filename = args.starting_model,
            model_type=args.model_type,
            model_architecture=args.architecture,
            # Training parameters
            optimizer_type=args.optimizer,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            rollout_length=args.rollout_length,
            reward_clip=args.reward_clip,
            reward_scale=args.reward_scale,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            norm_adv=args.norm_adv,
            clip_vf_loss=args.clip_vf_loss,
            vf_loss_coeff=args.vf_loss_coeff,
            entropy_loss_coeff=args.entropy_loss_coeff,
            target_kl=args.target_kl,
            minibatch_size=args.minibatch_size,
            num_minibatches=args.num_minibatches,
            max_grad_norm=args.max_grad_norm,
            cuda=args.cuda,
            load_optimizer=args.load_optimizer,
            # Plasticity experiment config
            steps_per_env = args.steps_per_env,
            num_cycles = args.num_cycles,
            cycle_task_stop_index = args.cycle_task_stop_index,
            num_tasks = args.num_tasks,
            # Override configs
            num_trials=args.num_trials,
            room_size=args.room_size,
    )
