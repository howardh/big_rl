import argparse
from collections import defaultdict
import itertools
import os

import numpy as np
import torch
import h5py
from torch.utils.data import default_collate
from tqdm import tqdm

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model
from big_rl.utils import merge_space
from big_rl.minigrid.hidden_info.hidden_info import OBJ_TO_IDX, get_all_objects_pos, get_all_objects_vector, get_relative_target_obj_pos, get_target_obj_idx, get_target_objs, get_relative_wall_map, OBJECTS


def count_obj_combinations(object_presence):
    count = set()
    for o in tqdm(object_presence):
        count.add(tuple(o))
    return len(count)


def run_episodes(model, env_config, preprocess_obs_fn, num_episodes, max_episode_length=None):
    env = make_env(**env_config)
    obj_pair_counts = np.eye(len(OBJECTS)) # Start with 1s on the diagonal because we never get pairs of the same object, so the diagonal never increases. We start the count at 1 so we can just check `obj_pair_counts.min()` to make sure all pairs are covered.

    for ep in tqdm(itertools.count(), total=num_episodes):
        episode_reward = 0 # total reward for the current episode
        episode_length = 0 # Number of transitions in the episode

        hidden = model.init_hidden(1) # type: ignore (???)

        obs, _ = env.reset()
        obs = preprocess_obs_fn(obs)

        # Keep a count of how many times each object pair appears
        # This is to ensure we get a good distribution of object pairs
        objs = [OBJ_TO_IDX[(o.color, o.type)] for o in env.objects] # type: ignore
        obj_pair_counts[objs[0], objs[1]] += 1
        obj_pair_counts[objs[1], objs[0]] += 1

        terminated = False
        truncated = False
        for t in itertools.count():
            if (terminated or truncated):
                break
            if max_episode_length is not None and t >= max_episode_length:
                truncated = True
                break
            with torch.no_grad():
                model_output = model(obs, hidden)

                hidden = model_output['hidden']

                action_probs = model_output['action'].softmax(1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().cpu().numpy().squeeze()

            obs, reward, terminated, truncated, info = env.step(action)
            obs = preprocess_obs_fn(obs)

            # Return the data as a lambda so we can delay the computation until we know we need it
            yield hidden, ep, lambda: {
                'targets': get_target_objs(env),
                'target_idx': get_target_obj_idx(env),
                'target_pos': get_relative_target_obj_pos(env),
                'wall_map': get_relative_wall_map(env, 25),
                'object_presence': get_all_objects_vector(env),
                'all_objects_pos': get_all_objects_pos(env),
            }

            episode_reward += float(reward)
            episode_length += 1
        tqdm.write(f'Episode {ep}\t {episode_length} steps\t {episode_reward:.2f} reward')
        tqdm.write(f'  Object pair counts: min {obj_pair_counts.min()}, max {obj_pair_counts.max()}, mean {obj_pair_counts.mean()}')


def preprocess_obs(obs):
    obs_scale = { 'obs (image)': 1.0 / 255.0 }
    obs_ignore = ['obs (mission)']
    return {
        k: torch.tensor(v, dtype=torch.float, device=device).unsqueeze(0)*obs_scale.get(k,1)
        for k,v in obs.items()
        if k not in obs_ignore
    }


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        help='Environments to test on.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model checkpoint to test.')
    parser.add_argument('--model-envs', type=str, nargs='*', default=None,
                        help='Environments whose observation spaces are used to initialize the model. If not specified, the "--env" environment will be used.')
    parser.add_argument('--num-episodes', type=int, default=1_000,
                        help='Number of episodes to draw samples from.')
    parser.add_argument('--max-episode-length', type=int, default=None,
                        help='Limit the length of each episode. For dev and debugging purposes.')
    parser.add_argument('--dataset-size', type=int, default=1_000_000,
                        help='Number of transitions to collect.')
    parser.add_argument('--require-all-combinations', action='store_true',
                        help='Require that all object combinations are present in the dataset.')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--output', type=str, default='./dataset.h5',
                        help='Path to a file to save dataset.')
    init_parser_model(parser)
    args = parser.parse_args()

    # Create environment
    ENV_CONFIG_PRESETS = env_config_presets()
    env_config = ENV_CONFIG_PRESETS[args.env]

    env = make_env(**env_config)
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with h5py.File(args.output, 'w') as dataset:
        # Get a sample to determine the shape of the dataset
        sample = next(run_episodes(model, env_config, preprocess_obs, 1))
        hidden_size = torch.cat([x.flatten() for x in sample[0]]).shape[0]
        hidden = dataset.create_dataset(
                'hidden', (args.dataset_size, hidden_size), dtype=np.float32)
        target_idx = dataset.create_dataset(
                'target_idx', (args.dataset_size, 1), dtype=np.int8)
        target_pos = dataset.create_dataset(
                'target_pos', (args.dataset_size, 2), dtype=np.int8)
        wall_map = dataset.create_dataset(
                'wall_map', (args.dataset_size, 25*25), dtype=np.int8)
        object_presence = dataset.create_dataset(
                'object_presence', (args.dataset_size, len(OBJECTS)), dtype=np.int8)
        all_objects_pos = dataset.create_dataset(
                'all_objects_pos', (args.dataset_size, 2*len(OBJECTS)), dtype=np.int8)

        obj_combination_count = defaultdict(lambda: 0)

        # Sample transitions and save them to the dataset
        last_ep = None
        for n,x in enumerate(run_episodes(model, env_config, preprocess_obs, args.num_episodes, max_episode_length=args.max_episode_length)):
            ep = x[1]
            # If we have reached the desired number of episodes ...
            if ep != last_ep and ep >= args.num_episodes:
                last_ep = ep
                # ... and if we have enough data, then consider whether we should terminate
                if n >= args.dataset_size:
                    # If we don't care about having all object combinations, we're done
                    if not args.require_all_combinations:
                        break
                    # Otherwise, make sure that we have at least one data point for every object pair
                    num_combinations = len(OBJECTS) * (len(OBJECTS) - 1) // 2
                    if len(obj_combination_count) >= num_combinations and min(obj_combination_count.values()) > 0:
                        break
                    tqdm.write(f'Not enough object combinations: {sum(x>0 for x in obj_combination_count.values())} / {num_combinations}')
                else:
                    # If we don't have enough data, then keep going past `args.num_episodes`
                    tqdm.write(f'Not enough data: {n} / {args.dataset_size}. Continuing past specified number of episodes.')

            # Uniformly sample a set of transitions from the stream of transitions
            if np.random.random() > min(1, args.dataset_size / (n+1)):
                continue
            if n < args.dataset_size:
                idx = n
            else:
                idx = np.random.randint(args.dataset_size)

            hidden_info = x[2]()
            hidden[idx] = torch.cat([x.flatten() for x in x[0]]).numpy()
            target_idx[idx] = hidden_info['target_idx']
            target_pos[idx] = hidden_info['target_pos']
            wall_map[idx] = hidden_info['wall_map'].flatten()
            all_objects_pos[idx] = hidden_info['all_objects_pos'].flatten()
            if object_presence[idx].sum() > 0:
                obj_combination_count[tuple(object_presence[idx])] -= 1
            object_presence[idx] = hidden_info['object_presence']
            obj_combination_count[tuple(object_presence[idx])] += 1
