import argparse
from collections import defaultdict
import os
import itertools
import json
import signal
import sys
from typing import Optional, Generator, Dict, List, Any, Callable, Union, Tuple, NamedTuple
import time
import uuid
import yaml

import gymnasium
import gymnasium.spaces
import gymnasium.vector
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn
import torch.nn.utils
import numpy as np
import wandb

from frankenstein.buffer.vec_history import VecHistoryBuffer # type: ignore
from frankenstein.advantage.gae import generalized_advantage_estimate # type: ignore
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss # type: ignore

from big_rl.minigrid.arguments import init_parser_trainer, init_parser_model
from big_rl.utils import torch_save, zip2, merge_space, generate_id
from big_rl.utils.make_env import EnvGroup, get_config_from_yaml, make_env_from_yaml#, make_env_labels_from_yaml, make_env_group_labels_from_yaml, make_env_to_model_mapping_from_yaml, make_eval_only_from_yaml
from big_rl.model import factory as model_factory
import big_rl.mujoco.envs # type: ignore (import for the env registration)
from big_rl.generic.evaluate_model import main as evaluate_model_main_, init_arg_parser as evaluate_model_init_arg_parser_
from big_rl.generic.common import to_tensor, reset_hidden, get_action_dist_function


WANDB_PROJECT_NAME = 'ppo-generic'


class Callbacks(NamedTuple):
    on_model_init: Callable[[Any], None] | None = None
    on_checkpoint_load: Callable[[Any], None] | None = None


def compute_ppo_losses(
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
        num_epochs : int,
        backtrack : bool,
        action_dist_fn : Callable) -> Generator[Dict[str,Union[torch.Tensor,Tuple[torch.Tensor,...]]],None,None]:
    """
    Compute the losses for PPO.
    """
    discount_pg = False
    if backtrack and target_kl is None:
        raise ValueError('`target_kl` must be specified when backtracking is enabled.')

    obs = history.obs
    action = history.action
    reward = history.reward
    terminal = history.terminal
    misc = history.misc
    assert isinstance(misc,dict)
    hidden = misc['hidden']
    time_step = misc['t']
    device = next(model.parameters()).device

    n = len(history.obs_history)
    num_training_envs = len(reward[0])
    initial_hidden = model.init_hidden(num_training_envs) # type: ignore

    with torch.no_grad():
        net_output = []
        curr_hidden = tuple([h[0].detach() for h in hidden])
        for o,term in zip2(obs,terminal):
            #curr_hidden = tuple([
            #    torch.where(
            #        term.view(-1, *([1]*(len(h.shape)-2))), # dim 1 is the number of modules, dim 2 is the batch size. The -2 is to skip those two, then we broadcast over the rest of the dimensions.
            #        init_h, h
            #    )
            #    for init_h,h in zip(initial_hidden,curr_hidden)
            #])
            curr_hidden = reset_hidden(
                    terminal = term,
                    hidden = curr_hidden,
                    initial_hidden = initial_hidden,
                    batch_dim = model.hidden_batch_dims,
            )
            #o = preprocess_input_fn(o) if preprocess_input_fn is not None else o
            o = to_tensor(o, device)
            no = model(o,curr_hidden)
            curr_hidden = no['hidden']
            net_output.append(no)
        net_output = default_collate(net_output)
        state_values_old = net_output['value'].squeeze(2)
        action_dist, action_dist_log_prob = action_dist_fn(net_output,n-1)
        log_action_probs_old = action_dist_log_prob(action) # FIXME: For continuous action spaces, this needs to be summed over the action size dimension.

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

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    state_dict = None

    for epoch in range(num_epochs):
        net_output = []
        curr_hidden = tuple([h[0].detach() for h in hidden])
        initial_hidden = model.init_hidden(num_training_envs) # type: ignore
        for o,term in zip2(obs,terminal):
            #curr_hidden = tuple([
            #    torch.where(term.view(-1, *([1]*(len(h.shape)-2))), init_h, h)
            #    for init_h,h in zip(initial_hidden,curr_hidden)
            #])
            curr_hidden = reset_hidden(
                    terminal = term,
                    hidden = curr_hidden,
                    initial_hidden = initial_hidden,
                    batch_dim = model.hidden_batch_dims,
            )
            #o = preprocess_input_fn(o) if preprocess_input_fn is not None else o
            o = to_tensor(o, device)
            no = model(o,curr_hidden)
            curr_hidden = no['hidden']
            net_output.append(no)
        net_output = default_collate(net_output)

        assert 'value' in net_output
        state_values = net_output['value'].squeeze()
        action_dist, action_dist_log_prob = action_dist_fn(net_output,n-1)
        log_action_probs = action_dist_log_prob(action)
        entropy = action_dist.entropy()

        with torch.no_grad():
            logratio = log_action_probs - log_action_probs_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()

        if target_kl is not None:
            if approx_kl > target_kl:
                if backtrack:
                    assert state_dict is not None
                    model.load_state_dict(state_dict)
                break

        # Policy loss
        pg_loss = clipped_advantage_policy_gradient_loss(
                log_action_probs = log_action_probs,
                old_log_action_probs = log_action_probs_old,
                advantages = advantages,
                terminals = terminal[:n-1],
                epsilon=0.1
        )
        if discount_pg:
            pg_loss *= discount ** time_step[:n-1]
        pg_loss = pg_loss.mean()

        # Value loss
        if clip_vf_loss is not None:
            v_loss_unclipped = (state_values[:n-1] - returns) ** 2
            v_clipped = state_values_old[:n-1] + torch.clamp(
                state_values[:n-1] - state_values_old[:n-1],
                -clip_vf_loss,
                clip_vf_loss,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((state_values[:n-1] - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - entropy_loss_coeff * entropy_loss + v_loss * vf_loss_coeff

        if backtrack:
            state_dict = model.state_dict()

        if not torch.isfinite(loss):
            raise ValueError('Invalid loss computed')

        yield {
                'loss': loss,
                'loss_pi': pg_loss,
                'loss_vf': v_loss,
                'loss_entropy': -entropy_loss,
                'approx_kl': approx_kl,
                'output': net_output,
                'hidden': tuple(h.detach() for h in curr_hidden),
                'epoch': torch.tensor(epoch),
        }


def log_episode_end(done, info, episode_reward, episode_true_reward, episode_reward_components, episode_steps, env_ids, env_label_to_id, global_step_counter):
    for env_label, env_id in env_label_to_id.items():
        done2 = done & (env_ids == env_id)
        if not done2.any():
            continue

        true_reward_source = 'accumulated during training'
        if 'episode' in info['final_info'][done2][0]:
            true_reward_source = 'RecordEpisodeStatistics wrapper'
            episode_true_reward = np.array([np.nan if x is None else x['episode']['r'].item() for x in info['final_info']])

        if wandb.run is not None:
            data = {
                    f'reward/{env_label}': episode_reward[done2].mean().item(),
                    f'true_reward/{env_label}': episode_true_reward[done2].mean().item(),
                    f'episode_length/{env_label}': episode_steps[done2].mean().item(),
                    'step': global_step_counter[0]-global_step_counter[1],
                    'step_total': global_step_counter[0],
            }

            # Energy
            if 'energy' in info['final_info'][0]:
                data['remaining_energy'] = np.mean([x['energy'] for x in info['final_info'][done2]])

            # Log individual reward components
            for k,v in episode_reward_components.items():
                data[f'reward_components/{env_label}/{k}'] = v[done2].mean().item()

            # Log the standard mujoco reward by summing up the components
            ANT_KEYS = ['reward_forward', 'reward_ctrl', 'reward_survive']
            if all(k in episode_reward_components.keys() for k in ANT_KEYS):
                standard_reward = sum(
                    episode_reward_components[k][done2].mean().item()
                    for k in ANT_KEYS
                )
                data[f'ant_standard_reward/{env_label}'] = standard_reward
                data[f'ant_standard_reward/all'] = standard_reward

            # Save data
            wandb.log(data, step = global_step_counter[0])
            wandb.summary['true_reward_source'] = true_reward_source
        print(f'  reward: {episode_true_reward[done2].mean():.2f}\t len: {episode_steps[done2].mean()} \t env: {env_label} ({done2.sum().item()})')


def train_single_env(
        global_step_counter: List[int],
        model: torch.nn.Module,
        env: gymnasium.vector.VectorEnv,
        env_labels: List[str],
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
        num_epochs: int = 4,
        target_kl: Optional[float] = None,
        norm_adv: bool = True,
        warmup_steps: int = 0,
        update_hidden_after_grad: bool = False,
        backtrack: bool = False,
        action_dist_fn: Callable | None = None,
        ) -> Generator[Dict[str, Any], None, None]:
    """
    Train a model with PPO on an Atari game.

    Args:
        model: ...
        env: `gym.vector.VectorEnv`
    """
    num_envs = env.num_envs

    if action_dist_fn is None:
        action_dist_fn = get_action_dist_function(env.single_action_space)

    env_label_to_id = {label: i for i,label in enumerate(set(env_labels))}
    env_ids = np.array([env_label_to_id[label] for label in env_labels])

    device = next(model.parameters()).device

    def preprocess_input(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
            for k,v in obs.items() if k not in obs_ignore
        }

    #assert isinstance(env.observation_space, gymnasium.spaces.Dict)
    #for k in obs_scale.keys():
    #    if k not in dict(env.observation_space):
    #        raise ValueError(f'observation space does not contain key {k}')

    history = VecHistoryBuffer(
            num_envs = num_envs,
            max_len=rollout_length+1,
            device=device)
    log = {}

    episode_true_reward = np.zeros(num_envs) # Actual reward we want to optimize before any modifications (e.g. clipping)
    episode_reward = np.zeros(num_envs) # Reward presented to the learning algorithm
    episode_reward_components = defaultdict(lambda: np.zeros(num_envs)) # The reward split between the various components (e.g. control cost, healthy reward, task reward, etc)
    episode_steps = np.zeros(num_envs)

    obs, info = env.reset()
    hidden = model.init_hidden(num_envs) # type: ignore (???)
    history.append_obs(
            #{k:v for k,v in obs.items() if k not in obs_ignore},
            obs,
            misc = {'hidden': hidden, 't': episode_steps.copy()},
    )

    ##################################################
    # Warmup
    # The state of all environments are similar at the start of an episode. By warming up, we increase the diversity of states to something that is closer to iid.

    start_time = time.time()
    warmup_episode_rewards = defaultdict(lambda: [])
    warmup_episode_steps = defaultdict(lambda: [])
    for _ in range(warmup_steps):
        # Select action
        with torch.no_grad():
            #model_output = model({
            #    k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
            #    for k,v in obs.items()
            #    if k not in obs_ignore
            #}, hidden)
            model_output = model(to_tensor(obs, device), hidden)
            hidden = model_output['hidden']

            action_dist, _ = action_dist_fn(model_output)
            action = action_dist.sample().cpu().numpy()
            breakpoint()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action) # type: ignore
        done = terminated | truncated

        episode_reward += reward
        episode_true_reward += info.get('reward', reward)
        episode_steps += 1

        if done.any():
            # Reset hidden state for finished episodes
            #hidden = tuple(
            #        torch.where(torch.tensor(done, device=device).unsqueeze(1), h0, h)
            #        for h0,h in zip(model.init_hidden(num_envs), hidden) # type: ignore (???)
            #)
            hidden = reset_hidden(
                    terminal = torch.tensor(done, device=device),
                    hidden = hidden,
                    initial_hidden = model.init_hidden(num_envs),
                    batch_dim = model.hidden_batch_dims,
            )
            # Print stats
            for env_label, env_id in env_label_to_id.items():
                done2 = done & (env_ids == env_id)
                if not done2.any():
                    continue
                for x in episode_reward[done2]:
                    warmup_episode_rewards[env_label].append(x.item())
                for x in episode_steps[done2]:
                    warmup_episode_steps[env_label].append(x.item())
                print(f'Warmup\t reward: {episode_reward[done2].mean():.2f}\t len: {episode_steps[done2].mean()} \t env: {env_label} ({done2.sum().item()})')
            # Reset episode stats
            episode_reward[done] = 0
            episode_true_reward[done] = 0
            episode_steps[done] = 0

    if warmup_steps > 0:
        print(f'Warmup time: {time.time() - start_time:.2f} s')
        for env_label, env_id in env_label_to_id.items():
            reward_mean = np.mean(warmup_episode_rewards[env_label])
            reward_std = np.std(warmup_episode_rewards[env_label])
            steps_mean = np.mean(warmup_episode_steps[env_label])
            # TODO: handle case where there are no completed episodes during warmup?
            if wandb.run is not None:
                wandb.log({
                    f'reward/{env_label}': reward_mean,
                    f'episode_length/{env_label}': steps_mean,
                    'step': global_step_counter[0]-global_step_counter[1],
                    'step_total': global_step_counter[0],
                }, step = global_step_counter[0])
            print(f'\t{env_label}\treward: {reward_mean:.2f} +/- {reward_std:.2f}\t len: {steps_mean:.2f}')

    ##################################################
    # Start training
    for step in itertools.count():
        # Gather data
        state_values = [] # For logging purposes
        entropies = [] # For logging purposes
        episode_rewards = [] # For multitask weighing purposes
        for _ in range(rollout_length):
            global_step_counter[0] += num_envs

            # Select action
            with torch.no_grad():
                #model_output = model({
                #    k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
                #    for k,v in obs.items()
                #    if k not in obs_ignore
                #}, hidden)
                model_output = model(to_tensor(obs, device), hidden)
                hidden = model_output['hidden']

                action_dist, _ = action_dist_fn(model_output)
                action = action_dist.sample().cpu().numpy()

                state_values.append(model_output['value'])
                entropies.append(action_dist.entropy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated | truncated

            history.append_action(action)
            episode_true_reward += info.get('reward', reward)
            episode_steps += 1
            for k in info.keys():
                if 'reward' in k and not k.startswith('_'):
                    episode_reward_components[k] += info[k]

            reward *= reward_scale
            if reward_clip is not None:
                reward = np.clip(reward, -reward_clip, reward_clip)

            episode_reward += reward # Sum up reward after clipping

            history.append_obs(
                    #{k:v for k,v in obs.items() if k not in obs_ignore}, reward, done,
                    obs, reward, done,
                    misc = {'hidden': hidden, 't': episode_steps.copy()}
            )

            if done.any():
                print(f'Episode finished ({step * num_envs * rollout_length:,} -- {global_step_counter[0]:,})')
                log_episode_end(
                        done = done,
                        info = info,
                        episode_reward = episode_reward,
                        episode_true_reward = episode_true_reward,
                        episode_reward_components = episode_reward_components,
                        episode_steps = episode_steps,
                        env_ids = env_ids,
                        env_label_to_id = env_label_to_id,
                        global_step_counter = global_step_counter,
                )
                # Reset hidden state for finished episodes
                #hidden = tuple(
                #        #torch.where(torch.tensor(done, device=device).unsqueeze(1), h0, h)
                #        torch.where(torch.tensor(done, device=device).view(-1, *([1]*(len(h.shape)-2))), h0, h)
                #        for h0,h in zip(model.init_hidden(num_envs), hidden) # type: ignore (???)
                #)
                hidden = reset_hidden(
                    terminal = torch.tensor(done, device=device),
                    hidden = hidden,
                    initial_hidden = model.init_hidden(num_envs),
                    batch_dim = model.hidden_batch_dims,
                )
                # Save episode rewards
                for r in episode_reward[done]:
                    episode_rewards.append(r)
                # Reset episode stats
                episode_reward[done] = 0
                episode_true_reward[done] = 0
                episode_steps[done] = 0
                for k in episode_reward_components.keys():
                    episode_reward_components[k][done] = 0

        if type(model).__name__ == 'ModularPolicy5':
            assert isinstance(model.last_attention, list)
            assert isinstance(model.last_input_labels, list)
            assert isinstance(model.last_output_attention, dict)
            log['attention max'] = {
                label: max([a.max().item() for a in attn])
                for label, attn in
                zip(model.last_input_labels,
                    zip(
                        model.last_attention[0].split(1,dim=2),
                        model.last_output_attention['action'].split(1,dim=1), # type: ignore (???)
                        model.last_output_attention['value'].split(1,dim=1), # type: ignore (???)
                    )
                )
            }

        # Train
        losses = compute_ppo_losses(
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
                num_epochs = num_epochs,
                backtrack = backtrack,
                action_dist_fn=action_dist_fn,
        )
        x = None
        for x in losses:
            log['state_value'] = torch.stack(state_values)
            log['entropy'] = torch.stack(entropies)

            yield {
                'log': log,
                'episode_rewards': episode_rewards,
                **x
            }

            log = {}

        # Clear data
        history.clear()
        episode_rewards = []

        # Update hidden state
        if x is not None and update_hidden_after_grad:
            history.misc_history[-1]['hidden'] = x['hidden']
            hidden = x['hidden']


def train(
        model: torch.nn.Module,
        envs: List[EnvGroup],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # XXX: Private class. This might break in the future.
        *,
        max_steps: int = 1000,
        max_steps_total: int = -1,
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
        num_epochs: int = 4,
        target_kl: Optional[float] = None,
        norm_adv: bool = True,
        warmup_steps: int = 0,
        update_hidden_after_grad: bool = False,
        backtrack: bool = False,
        start_step: int = 0,
        ):
    global_step_counter = [start_step, start_step]
    if max_steps_total > 0 and start_step > max_steps_total:
        print(f'Start step ({start_step}) is greater than max_steps_total ({max_steps_total}). Exiting.')
        return

    # Check that there is at least one environment to train on
    if not any(env.train_enabled for env in envs):
        print('No environments are enabled for training. Exiting.')
        return

    trainers = [
        train_single_env(
            global_step_counter = global_step_counter,
            model = model[env.model_name] if env.model_name is not None else model, # type: ignore
            env = env.env,
            env_labels = env.env_labels,
            obs_scale = obs_scale,
            rollout_length = rollout_length,
            reward_scale = reward_scale,
            reward_clip = reward_clip,
            discount = discount,
            gae_lambda = gae_lambda,
            clip_vf_loss = clip_vf_loss,
            entropy_loss_coeff = entropy_loss_coeff,
            vf_loss_coeff = vf_loss_coeff,
            num_epochs = num_epochs,
            target_kl = target_kl,
            norm_adv = norm_adv,
            warmup_steps = warmup_steps,
            update_hidden_after_grad = update_hidden_after_grad,
            backtrack = backtrack,
        )
        for env in envs
        if env.train_enabled
    ]

    # TODO: Eval envs
    #evaluators = [... for env in envs if env.eval_enabled]

    start_time = time.time()
    for _, losses in enumerate(zip(*trainers)):
        #env_steps = training_steps * rollout_length * sum(env.num_envs for env in envs)
        env_steps = global_step_counter[0]

        if max_steps > 0 and env_steps-start_step >= max_steps:
            print('Reached max steps')
            break
        if max_steps_total > 0 and env_steps >= max_steps_total:
            print('Reached total max steps')
            break

        mean_loss = torch.stack([x['loss'] for x in losses]).mean()
        optimizer.zero_grad()
        mean_loss.backward()
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
            for env,x in zip(envs, losses):
                label = env.name
                data = {
                    f'loss/pi/{label}': x['loss_pi'].item(),
                    f'loss/v/{label}': x['loss_vf'].item(),
                    f'loss/entropy/{label}': x['loss_entropy'].item(),
                    f'loss/total/{label}': x['loss'].item(),
                    f'approx_kl/{label}': x['approx_kl'].item(),
                    f'state_value/{label}': x['log']['state_value'].mean().item(),
                    f'entropy/{label}': x['log']['entropy'].mean().item(),
                    #last_approx_kl=approx_kl.item(),
                    #'learning_rate': lr_scheduler.get_lr()[0],
                    'epoch': x['epoch'],
                    'step': global_step_counter[0]-global_step_counter[1],
                    'step_total': global_step_counter[0],
                }
                if 'attention max' in x['log']:
                    for k,v in x['log']['attention max'].items():
                        data[f'attention max/{k}'] = v
                wandb.log(data, step = global_step_counter[0])

        yield {
            'step': global_step_counter[0],
        }

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Timing
        if env_steps > 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = (env_steps - start_step) / elapsed_time
            if max_steps > 0: # FIXME: I think this one is being calculated wrong
                remaining_time = int((max_steps - env_steps) / steps_per_sec)
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                remaining_seconds = (remaining_time % 3600) % 60
                print(f"Step {env_steps-start_step:,}/{max_steps:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
            elif max_steps_total > 0:
                remaining_time = int((max_steps_total - env_steps) / steps_per_sec)
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                remaining_seconds = (remaining_time % 3600) % 60
                print(f"Step {env_steps:,}/{max_steps_total:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
            else:
                elapsed_time = int(elapsed_time)
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                elapsed_seconds = (elapsed_time % 3600) % 60
                print(f"Step {env_steps-start_step:,} \t {int(steps_per_sec):,} SPS \t Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")


def evaluate(model: torch.nn.Module, env_config, model_config, num_episodes: int, results_filename: str | None, tempdir: str):
    os.makedirs(tempdir, exist_ok=True)
    checkpoint_filename = os.path.join(tempdir, f'model-{uuid.uuid4()}.pt')
    torch.save({'model': model.state_dict()}, checkpoint_filename)

    argparser = evaluate_model_init_arg_parser_()
    args = [
        '--env-config', env_config,
        '--model-config', model_config,
        '--model', checkpoint_filename,
        '--num-episodes', str(num_episodes),
        '--no-video',
    ]
    if results_filename is not None:
        args.extend(['--results', results_filename])
    evaluate_model_main_(argparser.parse_args(args))


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-id', type=str, default=None,
                        help='Identifier for the current experiment. This value is used for setting the "{RUN_ID}" variable. If not specified, then either a slurm job ID is used, or the current date and time, depending on what is available. Note that the time-based ID has a resolution of 1 second, so if two runs are started within less than a second of each other, they may end up with the same ID.')

    parser.add_argument('--envs', type=str, default=['ALE/Pong-v5'], nargs='*', help='Environments to train on')
    parser.add_argument('--num-envs', type=int, default=[16], nargs='*',
                        help='Number of environments to train on. If a single number is specified, it will be used for all environments. If a list of numbers is specified, it must have the same length as --env.')
    parser.add_argument('--env-labels', type=str, default=None, nargs='*', help='')
    parser.add_argument('--env-config', type=str, default=None,
                        help='Path to an environment config file (yaml format). If specified, all other environment arguments are ignored.')

    init_parser_trainer(parser)
    init_parser_model(parser)

    parser.add_argument('--starting-model', type=str, default=None,
                        help='Path to a model checkpoint to start training from.')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                        help='Path to a model checkpoint to save the model to.')
    parser.add_argument('--checkpoint-interval', type=int, default=1_000_000,
                        help='Number of transitions between checkpoints.')
    parser.add_argument('--eval-interval', type=int, default=1_000_000,
                        help='Number of transitions between evaluations.')
    parser.add_argument('--eval-episodes', type=int, default=0,
                        help='Number of episodes to evaluate the model for.')
    parser.add_argument('--eval-results-dir', type=str, default=None,
                        help='Directory in which to save evaluation results. If not specified, then the results are not saved.')
    parser.add_argument('--ignore-checkpoint-optimizer', action='store_true',
                        help='If set, then the optimizer state is not loaded from the checkpoint.')
    parser.add_argument('--step-count-label', type=str, default=None,
                        help='Label to use for the step count saved in the checkpoint. This is useful if we want to continue training from a checkpoint, but want to start the step count from a different value (e.g. for the purposes of `--max-step` or `--max-step-total`).')
    parser.add_argument('--sync-vector-env', action='store_true',
                        help='If set, then the vectorized environments are synchronized. This is useful for debugging, but is likely slower, so it should not be used for training.')

    parser.add_argument('--slurm-split', action='store_true', help='Set this flag to let the script know it is running on a SLURM cluster with one job split across an array job. This ensures that the same checkpoint is used for each of these jobs.')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true', help='Save results to W&B.')
    parser.add_argument('--wandb-id', type=str, default=None,
                        help='W&B run ID. Defaults to the run ID.')
    parser.add_argument('--wandb-project', type=str,
                        default=WANDB_PROJECT_NAME,
                        help='W&B project name.')
    parser.add_argument('--wandb-group', type=str, default='null',
                        help='JSON data used to group runs together in W&B.')

    return parser


def main(args, callbacks=Callbacks()):
    # Post-process string arguments
    # Note: Only non-string arguments and args.run_id can be used until the post-processing is done.
    if args.run_id is None:
        args.run_id = generate_id(slurm_split = args.slurm_split)
    try:
        args.wandb_group = json.loads(args.wandb_group)
    except:
        pass # If it's invalid json, then just treat it as a string. We're not doing anything with this besides passing it to wandb.config.update().

    substitutions = dict(
        RUN_ID = args.run_id,
        SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', None),
        SLURM_ARRAY_JOB_ID = os.environ.get('SLURM_ARRAY_JOB_ID', None),
        SLURM_ARRAY_TASK_ID = os.environ.get('SLURM_ARRAY_TASK_ID', None),
        SLURM_TASK_ID = os.environ.get('SLURM_PROCID', None),
        SLURM_STEP_ID = os.environ.get('SLURM_STEP_ID', None),
    )
    substitutions = {k:v.format(**substitutions) for k,v in substitutions.items() if v is not None} # Substitute values in `subsitutions` too in case they also have {} variables. None values are removed.
    for k,v in vars(args).items():
        if type(v) is str:
            v = v.format(**substitutions)
            setattr(args, k, v)

    print(substitutions)
    print('-'*80)
    print(f'Run ID: {args.run_id}')
    print(f'W&B ID: {args.wandb_id}')

    # Initialize W&B
    if args.wandb:
        if args.wandb_id is not None:
            wandb.init(project=args.wandb_project, id=args.wandb_id, resume='allow')
        elif args.model_checkpoint is not None: # XXX: Questionable decision. Maybe it should be explicitly specified as a CLI argument if we want to use the same W&B ID. This is causing more problems than it solves.
            wandb_id = os.path.basename(args.model_checkpoint).split('.')[0]
            wandb.init(project=args.wandb_project, id=wandb_id, resume='allow')
        else:
            wandb.init(project=args.wandb_project)
        wandb.config.update({k:v for k,v in args.__dict__.items() if k != 'model_config'})

    # Initialize environments
    if args.env_config is not None:
        envs = make_env_from_yaml(args.env_config)
    else:
        raise Exception('Environments must be configured via a config file.')

    # Device
    print('-'*80)
    if args.cuda and torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    # Initialize model
    if args.model_config is not None:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        if wandb.run is not None:
            wandb.config.update({'model_config': model_config}, allow_val_change=True)
        #observation_spaces = defaultdict(lambda: [])
        #action_spaces = defaultdict(lambda: [])
        #for env in envs:
        #    if env.model_name is None:
        #        continue
        #    observation_spaces[env.model_name].append(env.env.single_observation_space)
        #    action_spaces[env.model_name].append(env.env.single_action_space)
        #observation_spaces = {k: merge_space(*v) for k,v in observation_spaces.items()}
        #action_spaces = {k: merge_space(*v) for k,v in action_spaces.items()}

        model = model_factory.create_model(
            model_config,
            #observation_space = observation_spaces,
            #action_space = action_spaces,
            envs = envs,
        )
    else:
        raise Exception('Model must be configured via a config file.')
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('-'*80)
    print('Model initialized')
    print(f'Number of parameters: {param_count:,}')

    if callbacks.on_model_init is not None:
        callbacks.on_model_init(locals())

    # Initialize optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # Initialize learning rate scheduler
    lr_scheduler = None # TODO

    # Load checkpoint
    step_count_key = 'step' if args.step_count_label is None else f'step ({args.step_count_label})'
    start_step = 0
    checkpoint = None
    is_resume = False # Useful info for callbacks

    if args.model_checkpoint is not None and os.path.exists(args.model_checkpoint) and os.stat(args.model_checkpoint).st_size > 0:
        assert not os.path.isdir(args.model_checkpoint), 'Model checkpoint must be a file, not a directory'
        # The checkpoint exists, which means the experiment already started. Resume the experiment instead of restarting it.
        # If the checkpoint file is empty, it means we just created the file and the experiment hasn't started yet.
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        print(f'Experiment already started. Loading checkpoint from {args.model_checkpoint}')
        is_resume = True
    elif args.starting_model is not None:
        checkpoint = torch.load(args.starting_model, map_location=device)
        print(f'Loading starting model from {args.starting_model}')
        is_resume = False
    print('Resume:', is_resume)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and not args.ignore_checkpoint_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_step = checkpoint.get(step_count_key, 0)
        print(f'Loaded checkpoint successfully. Starting from step {start_step}.')

    if checkpoint is None:
        all_step_counts = {'step': 0}
    else:
        all_step_counts = {k: v for k,v in checkpoint.items() if k.startswith('step')}

    if callbacks.on_checkpoint_load is not None:
        callbacks.on_checkpoint_load(locals())

    # Initialize trainer
    print('-'*80)
    trainer = train(
            model = model,
            envs = envs,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            max_steps = args.max_steps,
            max_steps_total = args.max_steps_total,
            rollout_length = args.rollout_length,
            #obs_scale = {'obs': 1.0/255.0},
            obs_scale = {},
            reward_clip = args.reward_clip,
            reward_scale = args.reward_scale,
            discount = args.discount,
            gae_lambda = args.gae_lambda,
            norm_adv = args.norm_adv,
            clip_vf_loss = args.clip_vf_loss,
            vf_loss_coeff = args.vf_loss_coeff,
            entropy_loss_coeff = args.entropy_loss_coeff,
            target_kl = args.target_kl,
            num_epochs = args.num_epochs,
            max_grad_norm = args.max_grad_norm,
            warmup_steps = args.warmup_steps,
            update_hidden_after_grad = args.update_hidden_after_grad,
            backtrack = args.backtrack,
            start_step = start_step,
    )

    # Run training loop
    x = None
    def save_checkpoint(filename=args.model_checkpoint): # Make this a separate function in case we end up here with the debugger and want to manually save a checkpoint
        if x is None:
            return # This occurs if no training has occured (e.g. if we start an experiment that was already completed)
        assert isinstance(x,dict)
        torch_save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            **all_step_counts,
            'step': all_step_counts['step'] + (x['step'] - start_step),
            step_count_key: x['step'],
        }, filename+'.tmp')
        os.replace(filename+'.tmp', filename) # POSIX requires that this is atomic
        print(f'Saved checkpoint to {os.path.abspath(filename)}')

    sigusr1_received = False
    def handle_sigusr1(signum, frame):
        nonlocal sigusr1_received
        sigusr1_received = True

    signal.signal(signal.SIGUSR1, handle_sigusr1) # Save checkpoint on Ctrl+C or when Slurm terminates the job

    eval_count = start_step // args.eval_interval
    tempdir = os.environ.get('SLURM_TMPDIR', '/tmp')
    tempdir = os.path.join(tempdir, uuid.uuid4().hex)
    def maybe_eval_model():
        """ Evaluate the model if it's time to do so. Otherwise do nothing. """
        nonlocal eval_count, x
        if args.eval_episodes <= 0:
            return
        step = x['step'] if x is not None else start_step
        if step - (eval_count - 1) * args.eval_interval < args.eval_interval: # We want to evaluate once at step 0, then every `args.eval_interval` steps after that
            return

        if args.eval_results_dir is None:
            results_filename = None
        else:
            results_filename = os.path.join(args.eval_results_dir, f'eval-{step}.pt')

        evaluate(
            model=model,
            env_config=args.env_config,
            model_config=args.model_config,
            num_episodes=args.eval_episodes,
            results_filename=results_filename,
            tempdir=tempdir,
        )
        eval_count += 1

    checkpoint_count = start_step // args.checkpoint_interval
    if args.model_checkpoint is not None:
        os.makedirs(os.path.dirname(args.model_checkpoint), exist_ok=True)
        if not args.model_checkpoint.endswith('.pt'):
            # TODO: If the path is a directory, generate a unique filename
            raise NotImplementedError()
        maybe_eval_model() # Evaluate the model at the start of training
        for x in trainer:
            if x['step'] - checkpoint_count * args.checkpoint_interval >= args.checkpoint_interval:
                save_checkpoint(args.model_checkpoint)
                checkpoint_count += 1
            if sigusr1_received:
                print('Received SIGUSR1. Saving checkpoint...')
                save_checkpoint(args.model_checkpoint)
                sys.exit(0)
            maybe_eval_model()
        save_checkpoint(args.model_checkpoint)
    else:
        try:
            maybe_eval_model()
            for x in trainer:
                maybe_eval_model()
        except KeyboardInterrupt:
            print('Interrupted')
            pass
        # Sometimes, we run an experiment without intending to save the model, but then change our mind later.
        if x is not None:
            print('Saving model')
            tmp_checkpoint = f'./temp-checkpoint.pt'
            save_checkpoint(tmp_checkpoint)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)
