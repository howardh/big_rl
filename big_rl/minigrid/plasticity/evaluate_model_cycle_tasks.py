"""
A script to evaluate the plasticity loss in the recurrent model setup.
This is done by running the model on multiple tasks in sequence without resetting the hidden state in between, and observing the peak performance each time a task is revisited.

The script is a little complicated because it was copied from another evaluation script with some things removed, but not reorganized.
"""

import argparse
import itertools

import gymnasium
import torch
import numpy as np
from tqdm import tqdm
from minigrid.core.constants import COLOR_NAMES
import wandb

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model
from big_rl.utils import merge_space


def test(model, env_configs, num_envs, preprocess_obs_fn, steps_per_env, num_cycles, num_tasks, verbose=False):
    envs = {
        k: gymnasium.vector.AsyncVectorEnv([
            lambda config=v: make_env(**config)
            for _ in range(num_envs)
        ])
        for k,v in env_configs.items()
    }
    env = CycleEnvs(envs=envs, steps_per_env=steps_per_env, num_cycles=num_cycles, num_tasks=num_tasks)

    hidden = model.init_hidden(num_envs) # type: ignore (???)
    steps_iterator = itertools.count()
    #steps_iterator = range(100+np.random.randint(50)); print('--- DEBUG MODE ---')
    #steps_iterator = range(10); print('--- DEBUG MODE ---')
    steps_iterator = tqdm(steps_iterator)

    obs, info = env.reset()
    obs = preprocess_obs_fn(obs)
    terminated = False
    truncated = False
    for step in steps_iterator:
        with torch.no_grad():
            model_output = model(obs, hidden)

            hidden = model_output['hidden']

            action_probs = model_output['action'].softmax(1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(action)
        obs = preprocess_obs_fn(obs)

        # Console output and logging
        cycle_envs_info = info['cycle_envs_info']
        env_label = cycle_envs_info["env_label"]
        env_index = cycle_envs_info["env_index"]
        if cycle_envs_info["env_change"] or step == 0:
            tqdm.write('-'*80)
            tqdm.write(f'Environment: {env_label}')
            tqdm.write('-'*80)
        if len(cycle_envs_info['episode_rewards']) > 0:
            if wandb.run is not None:
                wandb.log({
                    'reward': np.mean(cycle_envs_info['episode_rewards']),
                    f'reward/{env_label}': np.mean(cycle_envs_info['episode_rewards']),
                    f'reward/by_index/{env_index}': np.mean(cycle_envs_info['episode_rewards']),

                    'episode_length': np.mean(cycle_envs_info['episode_lengths']),
                    f'episode_length/{env_label}': np.mean(cycle_envs_info['episode_lengths']),
                    f'episode_length/by_index/{env_index}': np.mean(cycle_envs_info['episode_lengths']),

                    f'env_step/{env_label}': cycle_envs_info['env_step'],
                    f'env_step/by_index/{env_index}': cycle_envs_info['env_step'],
                }, step=cycle_envs_info['total_steps'])
            for ep_len,ep_rew in zip(cycle_envs_info['episode_lengths'],cycle_envs_info['episode_rewards']):
                tqdm.write(f"{cycle_envs_info['total_steps']:,} steps\t Episode length: {ep_len}, episode reward: {ep_rew}")

        if terminated or truncated:
            break


def split_results_by_target(results):
    """ Split up the results by target object. This is used when the target is randomized in the middle of an episode to get information on how well the agent performs for each target change. """

    assert 'target' in results
    if len(results['target']) == 0:
        return []

    keys = ['reward']

    current_target = None
    current_results = {}
    split_results = []
    for i, target in enumerate(results['target']):
        if target != current_target:
            if current_target is not None:
                split_results.append((current_target,current_results))
            current_target = target
            current_results = {k: [] for k in keys}
        for k in keys:
            current_results[k].append(results[k][i])
    split_results.append((current_target,current_results))
    return split_results


def preprocess_obs(obs):
    obs_scale = { 'obs (image)': 1.0 / 255.0 }
    obs_ignore = ['obs (mission)']
    return {
        k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
        for k,v in obs.items()
        if k not in obs_ignore
    }


class CycleEnvs(gymnasium.Env):
    def __init__(self, envs: dict, steps_per_env, num_cycles, num_tasks):
        self.envs = envs
        self.steps_per_env = steps_per_env
        self.num_cycles = num_cycles
        self.num_tasks = num_tasks

        self.num_envs = next(iter(envs.values())).num_envs
        for env in envs.values():
            if env.num_envs != self.num_envs:
                raise ValueError("All envs must have the same number of environments")

        self._total_steps = 0 # Total number of transitions
        self._current_step = 0 # Number of steps in the current environment in the current cycle
        self._env_change = False # Whether the environment has changed in the last step. The flag needs to be saved because it is sent on the next step.
        self._env_step = {k: 0 for k in self.envs.keys()} # Number of steps in each environment summed over all cycles
        self._episode_length = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_reward = np.zeros(self.num_envs, dtype=np.float32)

    def reset(self):
        self._env_order = list(self.envs.keys())
        np.random.shuffle(self._env_order)
        self._env_order = self._env_order[:self.num_tasks]
        print(f"Environment order: {self._env_order}")

        self._total_steps = 0
        self._index = 0
        self._current_step = 0
        self._env_change = False
        self._env_step = {k: 0 for k in self.envs.keys()}
        self._episode_length *= 0
        self._episode_reward *= 0

        obs, info = self.current_env.reset()
        done = np.zeros(self.num_envs, dtype=bool)
        if 'cycle_envs_info' in info:
            raise ValueError("Key conflict. 'cycle_envs_info' already in info.")
        info['cycle_envs_info'] = {
                'env_label': self.current_env_label,
                'env_index': self.current_env_index,
                'episode_rewards': self._episode_reward[done],
                'episode_lengths': self._episode_length[done],
                'env_change': False,
                'env_step': self._env_step[self.current_env_label],
                'total_steps': self._total_steps,
        }
        return obs, info

    @property
    def current_cycle(self):
        return self._index // len(self._env_order)

    @property
    def current_env_index(self):
        return self._index % len(self._env_order)

    @property
    def current_env_label(self):
        return self._env_order[self.current_env_index]

    @property
    def current_env(self):
        return self.envs[self.current_env_label]

    def step(self, action):
        self._total_steps += self.num_envs
        self._env_step[self.current_env_label] += self.num_envs
        self._current_step += self.num_envs
        terminated = False
        truncated = False
        env_change = False

        obs, reward, terminated_, truncated_, info = self.current_env.step(action)
        done = terminated_ | truncated_

        self._episode_reward += reward
        self._episode_length += 1

        if 'cycle_envs_info' in info:
            raise ValueError("Key conflict. 'cycle_envs_info' already in info.")
        info['cycle_envs_info'] = {
                'env_label': self.current_env_label,
                'env_index': self.current_env_index,
                'episode_rewards': self._episode_reward[done],
                'episode_lengths': self._episode_length[done],
                'env_change': self._env_change,
                'env_step': self._env_step[self.current_env_label],
                'total_steps': self._total_steps,
        }

        self._episode_reward[done] = 0
        self._episode_length[done] = 0

        # If we've run enough steps on this environment, move to the next one
        self._env_change = False
        if self._current_step >= self.steps_per_env:
            self._env_change = True
            self._index += 1
            self._current_step = 0

            self._episode_reward *= 0
            self._episode_length *= 0

            if self.current_cycle >= self.num_cycles:
                terminated = True
                truncated = False

            # Ignore the first observation
            # Return the obs from the current env so the agent gets the reward information
            _, _ = self.current_env.reset()

        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
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

    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model checkpoint to test.')
    parser.add_argument('--model-envs', type=str, nargs='*', default=None,
                        help='Environments whose observation spaces are used to initialize the model. If not specified, the first "--envs" environment will be used.')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--results', type=str, default=None,
                        help='Path to a file to save results.')
    parser.add_argument('--wandb', action='store_true', help='Save results to W&B.')
    parser.add_argument('--wandb-id', type=str, default=None,
                        help='W&B run ID.')
    init_parser_model(parser)
    args = parser.parse_args()

    # Create environment
    ENV_CONFIG_PRESETS = env_config_presets()
    if args.env_labels is None:
        args.env_labels = args.envs
    env_configs = {
            env_label: ENV_CONFIG_PRESETS[env_config_name]
            for env_config_name, env_label in zip(args.envs, args.env_labels)
    }

    env = make_env(**next(iter(env_configs.values())))
    if args.model_envs is not None:
        dummy_envs = [
                make_env(**ENV_CONFIG_PRESETS[c]) for c in args.model_envs]
    else:
        dummy_envs = [env]
    observation_space = merge_space(*[e.observation_space for e in dummy_envs])

    # Load model
    device = torch.device('cpu')
    model = init_model(
            observation_space = observation_space,
            action_space = env.action_space,
            model_type = args.model_type,
            recurrence_type = args.recurrence_type,
            architecture = args.architecture,
            hidden_size = args.hidden_size,
            device = device,
    )
    model.to(device)

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f'Loaded checkpoint from {args.model}')
    else:
        print('No model checkpoint specified, using random initialization.')
        checkpoint = {}

    if 'step' in checkpoint:
        print(f'Checkpoint step: {checkpoint["step"]:,}')

    if args.wandb:
        if args.wandb_id is not None:
            wandb.init(project='big_rl-metarl-plasticity', id=args.wandb_id, resume='allow')
        else:
            wandb.init(project='big_rl-metarl-plasticity')
        wandb.config.update(args, allow_val_change=True)

    # Test model
    test(
        model=model,
        env_configs=env_configs,
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        num_cycles=args.num_cycles,
        num_tasks=args.num_tasks,
        preprocess_obs_fn=preprocess_obs,
        verbose=args.verbose
    )
