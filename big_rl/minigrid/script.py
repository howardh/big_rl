import itertools
from typing import Optional, Generator, Dict, List, Any, Sequence, Iterable, Union, Tuple, Mapping, Callable
import time

from torchtyping import TensorType
import gymnasium
import gymnasium.spaces
import gymnasium.vector
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import wandb

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss

from big_rl.model.model import ModularPolicy2, ModularPolicy4, ModularPolicy5
from big_rl.minigrid.envs import make_env


def init_model(observation_space, action_space,
        model_type,
        recurrence_type,
        num_recurrence_blocks=3,
        architecture=[3,3],
        device=torch.device('cpu')):
    observation_space = observation_space # Unused variable
    inputs = {
        'obs (image)': {
            'type': 'ImageInput56',
            'config': {
                'in_channels': observation_space['obs (image)'].shape[0]
            },
        },
        'reward': {
            'type': 'ScalarInput',
        },
        'action': {
            'type': 'DiscreteInput',
            'config': {
                'input_size': action_space.n
            },
        },
    }
    if 'obs (reward_permutation)' in observation_space.keys():
        inputs['obs (reward_permutation)'] = {
            'type': 'LinearInput',
            'config': {
                'input_size': observation_space['obs (reward_permutation)'].shape[0]
            }
        }
    if 'action_map' in observation_space.keys():
        inputs['action_map'] = {
            'type': 'MatrixInput',
            'config': {
                'input_size': list(observation_space['action_map'].shape),
                'num_heads': 8,
            }
        }
    outputs = {
        'value': {
            'type': 'LinearOutput',
            'config': {
                'output_size': 1,
            }
        },
        'action': {
            'type': 'LinearOutput',
            'config': {
                'output_size': action_space.n,
            }
        },
    }
    common_model_params = {
        'inputs': inputs,
        'outputs': outputs,
        'input_size': 512,
        'key_size': 512,
        'value_size': 512,
        'num_heads': 8,
        'ff_size': 1024,
        'recurrence_type': recurrence_type,
    }
    if model_type == 'ModularPolicy2':
        return ModularPolicy2(
                **common_model_params,
                num_blocks=num_recurrence_blocks,
        ).to(device)
    elif model_type == 'ModularPolicy4':
        assert architecture is not None
        return ModularPolicy4(
                **common_model_params,
                architecture=architecture,
        ).to(device)
    elif model_type == 'ModularPolicy5':
        assert architecture is not None
        return ModularPolicy5(
                **common_model_params,
                architecture=architecture,
        ).to(device)
    raise NotImplementedError()


def merge_space(*spaces):
    new_space = {}
    for space in spaces:
        for k,v in space.items():
            if k in new_space:
                assert new_space[k] == v, f"Space mismatch for key {k}: {new_space[k]} != {v}"
            else:
                new_space[k] = v
    return gymnasium.spaces.Dict(new_space)


def zip2(*args) -> Iterable[Union[Tuple,Mapping]]:
    """
    Zip objects together. If dictionaries are provided, the lists within the dictionary are zipped together.

    >>> list(zip2([1,2,3], [4,5,6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(zip2({'a': [4,5,6], 'b': [7,8,9]}))
    [{'a': 4, 'b': 7}, {'a': 5, 'b': 8}, {'a': 6, 'b': 9}]

    >>> list(zip2([1,2,3], {'a': [4,5,6], 'b': [7,8,9]}))
    [(1, {'a': 4, 'b': 7}), (2, {'a': 5, 'b': 8}), (3, {'a': 6, 'b': 9})]

    >>> import torch
    >>> list(zip2(torch.tensor([1,2,3]), torch.tensor([4,5,6])))
    [(tensor(1), tensor(4)), (tensor(2), tensor(5)), (tensor(3), tensor(6))]
    """
    if len(args) == 1:
        if isinstance(args[0],(Sequence)):
            return args[0]
        if isinstance(args[0],torch.Tensor):
            return (x for x in args[0])
        if isinstance(args[0], dict):
            keys = args[0].keys()
            return (dict(zip(keys, vals)) for vals in zip(*(args[0][k] for k in keys)))
    return zip(*[zip2(a) for a in args])


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
        num_epochs : int) -> Generator[Dict[str,TensorType],None,None]:
    """
    Compute the losses for PPO.
    """
    obs = history.obs
    action = history.action
    reward = history.reward
    terminal = history.terminal
    misc = history.misc
    assert isinstance(misc,dict)
    hidden = misc['hidden']

    n = len(history.obs_history)
    num_training_envs = len(reward[0])
    initial_hidden = model.init_hidden(num_training_envs) # type: ignore

    with torch.no_grad():
        net_output = []
        curr_hidden = tuple([h[0].detach() for h in hidden])
        for o,term in zip2(obs,terminal):
            curr_hidden = tuple([
                torch.where(term.unsqueeze(1), init_h, h)
                for init_h,h in zip(initial_hidden,curr_hidden)
            ])
            o = preprocess_input_fn(o) if preprocess_input_fn is not None else o
            no = model(o,curr_hidden)
            curr_hidden = no['hidden']
            net_output.append(no)
        net_output = default_collate(net_output)
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

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(num_epochs):
        net_output = []
        curr_hidden = tuple([h[0].detach() for h in hidden])
        for o,term in zip2(obs,terminal):
            curr_hidden = tuple([
                torch.where(term.unsqueeze(1), init_h, h)
                for init_h,h in zip(initial_hidden,curr_hidden)
            ])
            o = preprocess_input_fn(o) if preprocess_input_fn is not None else o
            no = model(o,curr_hidden)
            curr_hidden = no['hidden']
            net_output.append(no)
        net_output = default_collate(net_output)

        assert 'value' in net_output
        assert 'action' in net_output
        state_values = net_output['value'].squeeze()
        action_dist = torch.distributions.Categorical(logits=net_output['action'][:n-1])
        log_action_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        with torch.no_grad():
            logratio = log_action_probs - log_action_probs_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss = clipped_advantage_policy_gradient_loss(
                log_action_probs = log_action_probs,
                old_log_action_probs = log_action_probs_old,
                advantages = advantages,
                terminals = terminal[:n-1],
                epsilon=0.1
        ).mean()

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

        yield {
                'loss': loss,
                'loss_pi': pg_loss,
                'loss_vf': v_loss,
                'loss_entropy': -entropy_loss,
                'approx_kl': approx_kl,
                'output': net_output,
        }

        if target_kl is not None:
            if approx_kl > target_kl:
                break


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
        ) -> Generator[Dict[str, Any], None, None]:
    """
    Train a model with PPO on an Atari game.

    Args:
        model: ...
        env: `gym.vector.VectorEnv`
    """
    num_envs = env.num_envs

    env_label_to_id = {label: i for i,label in enumerate(set(env_labels))}
    env_ids = np.array([env_label_to_id[label] for label in env_labels])

    device = next(model.parameters()).device

    def preprocess_input(obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            k: torch.tensor(v, device=device)*obs_scale.get(k,1)
            for k,v in obs.items() if k not in obs_ignore
        }

    history = VecHistoryBuffer(
            num_envs = num_envs,
            max_len=rollout_length+1,
            device=device)

    obs, info = env.reset()
    hidden = model.init_hidden(num_envs) # type: ignore (???)
    history.append_obs(
            {k:v for k,v in obs.items() if k not in obs_ignore},
            misc = {'hidden': hidden},
    )
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    for step in itertools.count():
        # Gather data
        state_values = [] # For logging purposes
        entropies = [] # For logging purposes
        for _ in range(rollout_length):
            global_step_counter[0] += num_envs

            # Select action
            with torch.no_grad():
                model_output = model({
                    k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
                    for k,v in obs.items()
                    if k not in obs_ignore
                }, hidden)
                hidden = model_output['hidden']

                action_probs = model_output['action'].softmax(1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().cpu().numpy()

                state_values.append(model_output['value'])
                entropies.append(action_dist.entropy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            history.append_action(action)
            episode_reward += reward
            episode_steps += 1

            reward *= reward_scale
            if reward_clip is not None:
                reward = np.clip(reward, -reward_clip, reward_clip)

            history.append_obs(
                    {k:v for k,v in obs.items() if k not in obs_ignore}, reward, done,
                    misc = {'hidden': hidden}
            )

            if done.any():
                if 'lives' in info:
                    done = done & (info['lives'] == 0)
            if done.any():
                print(f'Episode finished ({step * num_envs * rollout_length:,})')
                for env_label, env_id in env_label_to_id.items():
                    done2 = done & (env_ids == env_id)
                    if not done2.any():
                        continue
                    if wandb.run is not None:
                        wandb.log({
                                f'reward/{env_label}': episode_reward[done2].mean().item(),
                                f'episode_length/{env_label}': episode_steps[done2].mean().item(),
                                'step': global_step_counter[0],
                        }, step = global_step_counter[0])
                    print(f'  reward: {episode_reward[done].mean():.2f}\t len: {episode_steps[done].mean()} \t env: {env_label} ({done2.sum().item()})')
                # Reset hidden state for finished episodes
                hidden = tuple(
                        torch.where(torch.tensor(done, device=device).unsqueeze(1), h0, h)
                        for h0,h in zip(model.init_hidden(num_envs), hidden) # type: ignore (???)
                )
                # Reset episode stats
                episode_reward[done] = 0
                episode_steps[done] = 0

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
        )
        for x in losses:
            yield {
                'state_value': torch.stack(state_values),
                'entropy': torch.stack(entropies),
                **x
            }

        # Clear data
        history.clear()


def train(
        model: torch.nn.Module,
        envs: List[gymnasium.vector.VectorEnv],
        env_labels: List[List[str]],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # XXX: Private class. This might break in the future.
        *,
        max_steps: int = 1000,
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
        ):
    global_step_counter = [0]
    trainers = [
        train_single_env(
            global_step_counter = global_step_counter,
            model = model,
            env = env,
            env_labels = labels,
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
        )
        for env,labels in zip(envs, env_labels)
    ]
    start_time = time.time()
    for _, losses in enumerate(zip(*trainers)):
        #env_steps = training_steps * rollout_length * sum(env.num_envs for env in envs)
        env_steps = global_step_counter[0]

        mean_loss = torch.stack([x['loss'] for x in losses]).mean()
        optimizer.zero_grad()
        mean_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if wandb.run is not None:
            for label,x in zip(env_labels, losses):
                wandb.log({
                    f'loss/pi/{label}': x['loss_pi'].item(),
                    f'loss/v/{label}': x['loss_vf'].item(),
                    f'loss/entropy/{label}': x['loss_entropy'].item(),
                    f'loss/total/{label}': x['loss'].item(),
                    f'approx_kl/{label}': x['approx_kl'].item(),
                    f'state_value/{label}': x['state_value'].mean().item(),
                    f'entropy/{label}': x['entropy'].mean().item(),
                    #last_approx_kl=approx_kl.item(),
                    #'learning_rate': lr_scheduler.get_lr()[0],
                    'step': global_step_counter[0],
                }, step = global_step_counter[0])

        yield

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Timing
        if env_steps > 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = env_steps / elapsed_time
            if max_steps > 0:
                remaining_time = int((max_steps - env_steps) / steps_per_sec)
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                remaining_seconds = (remaining_time % 3600) % 60
                print(f"Step {env_steps:,}/{max_steps:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
            else:
                elapsed_time = int(elapsed_time)
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                elapsed_seconds = (elapsed_time % 3600) % 60
                print(f"Step {env_steps:,} \t {int(steps_per_sec):,} SPS \t Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")


def env_config_presets():
    return {
        'fetch-001': {
            'env_name': 'MiniGrid-MultiRoom-v1',
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 5,
                'max_room_size': 5,
                'door_prob': 0.5,
                'max_steps_multiplier': 5,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_types': 1,
                    'num_obj_colors': 2,
                    'prob': 1.0, # 0.0 chance of flipping the reward
                },
                #'task_randomization_prob': 0.02, # 86% chance of happening at least once, with a 50% change of the randomized task being unchanged.
            }
        },

        'delayed-001': {
            'env_name': 'MiniGrid-Delayed-Reward-v0',
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 5,
                'max_room_size': 6,
                'door_prob': 0.5,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_colors': 6,
                    'prob': 1.0, # 0.0 chance of flipping the reward
                },
                #'task_randomization_prob': 0.02, # 86% chance of happening at least once, with a 50% change of the randomized task being unchanged.
            }
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--envs', type=str, default=['fetch-001'], nargs='*', help='Environments to train on')
    parser.add_argument('--num-envs', type=int, default=[16], nargs='*',
            help='Number of environments to train on. If a single number is specified, it will be used for all environments. If a list of numbers is specified, it must have the same length as --env.')
    parser.add_argument('--env-labels', type=str, default=None, nargs='*', help='')

    parser.add_argument('--max-steps', type=int, default=0, help='Number of training steps to run. One step is one weight update.')

    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer', choices=['Adam', 'RMSprop'])
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')

    parser.add_argument('--rollout-length', type=int, default=128, help='Length of rollout.')
    parser.add_argument('--reward-clip', type=float, default=1, help='Clip the reward magnitude to this value.')
    parser.add_argument('--reward-scale', type=float, default=1, help='Scale the reward magnitude by this value.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for GAE.')
    parser.add_argument('--norm-adv', type=bool, default=True, help='Normalize the advantages.')
    parser.add_argument('--clip-vf-loss', type=float, default=None, help='Clip the value function loss.')
    parser.add_argument('--vf-loss-coeff', type=float, default=0.5, help='Coefficient for the value function loss.')
    parser.add_argument('--entropy-loss-coeff', type=float, default=0.01, help='Coefficient for the entropy loss.')
    parser.add_argument('--target-kl', type=float, default=None, help='Target KL divergence.')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of minibatches.')
    parser.add_argument('--max-grad-norm', type=float, default=None, help='Maximum gradient norm.')

    parser.add_argument('--model-type', type=str, default='ModularPolicy5', help='Model type', choices=['ModularPolicy5'])
    parser.add_argument('--recurrence-type', type=str, default='RecurrentAttention11', help='Recurrence type', choices=[f'RecurrentAttention{i}' for i in [11,14]])
    parser.add_argument('--architecture', type=int, default=[3,3], nargs='*', help='Size of each layer in the model\'s core')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true', help='Save results to W&B.')

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='ppo-multitask-minigrid')
        wandb.config.update(args)

    #num_envs = args.num_envs
    #if len(num_envs) == 1:
    #    num_envs = num_envs * len(args.env)
    #env_name_to_label = {}
    #if args.env_labels is None:
    #    env_name_to_labels = {n: n for n in args.env}
    #else:
    #    env_name_to_labels = {n: l for n, l in zip(args.env, args.env_labels)}
    #env_names = list(itertools.chain(*[[e] * n for e, n in zip(args.env, num_envs)]))
    #env_labels = [env_name_to_labels[e] for e in env_names]

    ENV_CONFIG_PRESETS = env_config_presets()
    env_configs = [
        [ENV_CONFIG_PRESETS[e] for _ in range(n)]
        for e,n in zip(args.envs, args.num_envs)
    ]
    envs = [
        gymnasium.vector.AsyncVectorEnv([lambda conf=conf: make_env(**conf) for conf in env_config]) # type: ignore (Why is `make_env` missing an argument?)
        for env_config in env_configs
    ]

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = init_model(
            observation_space = merge_space(*[env.single_observation_space for env in envs]),
            action_space = envs[0].single_action_space, # Assume the same action space for all environments
            model_type = args.model_type,
            recurrence_type = args.recurrence_type,
            architecture = args.architecture,
            device = device,
    )
    model.to(device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    lr_scheduler = None # TODO

    trainer = train(
            model = model,
            envs = envs, # type: ignore (??? AsyncVectorEnv is not a subtype of VectorEnv ???)
            env_labels = [['doot']*len(envs)], # TODO: Make this configurable
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            max_steps = args.max_steps,
            rollout_length = args.rollout_length,
            obs_scale = {'obs (image)': 1.0/255.0},
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
    )
    for _ in trainer:
        pass
