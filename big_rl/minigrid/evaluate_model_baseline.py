import argparse
import itertools
import os
from typing import Tuple, Optional, Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
from fonts.ttf import Roboto # type: ignore

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model
from big_rl.utils import merge_space


def test(model, env_config, preprocess_obs_fn, video_callback_fn=None, verbose=False):
    env = make_env(**env_config)

    episode_reward = 0 # total reward for the current episode
    episode_length = 0 # Number of transitions in the episode
    results: dict[str,Any] = {
            'reward': [],
            'regret': [],
            'attention': [],
            'input_labels': [],
            'target': [], # Goal string
            'shaped_reward': [], # Predicted shaped reward
    }

    steps_iterator = itertools.count()
    #steps_iterator = range(100+np.random.randint(50)); print('--- DEBUG MODE ---')
    #steps_iterator = range(10); print('--- DEBUG MODE ---')
    if verbose:
        steps_iterator = tqdm(steps_iterator)

    obs, info = env.reset()
    obs = preprocess_obs_fn(obs)
    terminated = False
    truncated = False
    for _ in steps_iterator:
        with torch.no_grad():
            model_output = model(obs)

            action_probs = model_output['action'].softmax(1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(action)
        obs = preprocess_obs_fn(obs)

        results['reward'].append(reward)
        results['regret'].append(info.get('regret', None))
        if model.has_attention:
            results['attention'].append((
                [x.numpy() for x in model.last_attention],
                [x.numpy() for x in model.last_ff_gating],
                {k: v.numpy() for k,v in model.last_output_attention.items()},
            ))
            results['shaped_reward'].append(compute_shaped_reward(model))
        results['input_labels'].append(model.last_input_labels)
        if hasattr(env, 'goal_str'):
            results['target'].append(env.goal_str) # type: ignore

        episode_reward += float(reward)
        episode_length += 1

        if verbose:
            if reward != 0:
                tqdm.write(f"Step {episode_length}: reward={reward} total={episode_reward}")

        if terminated or truncated:
            break

        if video_callback_fn is not None:
            video_callback_fn(model, env, obs, results)

    if 'supervised_trials' in info:
        results['supervision'] = {}
        results['supervision']['supervised_trials'] = info['supervised_trials']
        results['supervision']['unsupervised_trials'] = info['unsupervised_trials']
        results['supervision']['supervised_reward'] = info['supervised_reward']
        results['supervision']['unsupervised_reward'] = info['unsupervised_reward']
    results['reward_by_trial'] = info['reward_by_trial']
    results['regret_by_trial'] = info['regret_by_trial']

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'results': results,
    }


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


def compute_shaped_reward(model):
    query = model.input_modules['obs (shaped_reward)'].key
    key = model.last_keys
    value = model.last_values
    attention = torch.einsum('k,bik->b', query, key).softmax(0)
    output = torch.einsum('b,bik->k', attention, value)
    return output.cpu().detach()


def concat_images(images, padding=0, direction='h', align=0):
    """
    Concatenate images along a given direction.

    Args:
        images: list of PIL images or callable functions that return PIL images. The functions must take two arguments: width and height.
    """
    if direction == 'h':
        width = sum([i.size[0] for i in images]) + padding * (len(images) + 1)
        height = max([i.size[1] for i in images]) + padding*2
        new_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
        x = 0
        for i in images:
            new_image.paste(i, (x+padding, (height - 2*padding - i.size[1]) // 2 * (align + 1) + padding))
            x += i.size[0] + padding
        return new_image
    elif direction == 'v':
        width = max([i.size[0] for i in images]) + padding*2
        height = sum([i.size[1] for i in images]) + padding * (len(images) + 1)
        new_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
        y = 0
        for i in images:
            new_image.paste(i, ((width - 2*padding - i.size[0]) // 2 * (align + 1) + padding, y + padding))
            y += i.size[1] + padding
        return new_image
    else:
        raise ValueError('direction must be "h" or "v"')

def draw_text(text, font_family=Roboto, font_size=18, color=(0,0,0), padding=2):
    font = PIL.ImageFont.truetype(font_family, font_size)

    text_width, text_height = font.getsize(text)
    img = PIL.Image.new('RGB',
            (text_width+2*padding, text_height+2*padding),
            color=(255,255,255))
    draw = PIL.ImageDraw.Draw(img)
    draw.fontmode = 'L' # type: ignore
    draw.text(
            (padding, padding),
            text,
            font=font,
            fill=color
    )
    return img

def draw_rewards(rewards: list, target_object):
    img1 = draw_text(
            'Reward: ' + ' '.join(f'{r}' for r in reversed(rewards[-5:])))
    img2 = draw_text(
            f'Total Reward: {sum(rewards):.2f}')
    img3 = draw_text(f'Target: {target_object}')
    return concat_images([img1, img2, img3], direction='v', align=-1)

def draw_observations(observations: dict):
    """ Draw reward, shaped reward, last action """
    images = [
            draw_text(f'{k}: {v.item()}')
            for k,v in observations.items()
            if k in ['reward', 'obs (shaped_reward)', 'action']
    ]
    return concat_images(images, direction='v', align=-1)


class VideoCallback:
    def __init__(self, filename: str, 
                 size: Optional[Tuple[int,int]] = None,
                 fps: int = 30):
        assert filename.endswith('.webm')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        self.filename = filename
        self.fps = fps

        if size is not None:
            self._video_writer = cv2.VideoWriter( # type: ignore
                    filename,
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    fps,
                    (env.envs[0].unwrapped.width*32, env.envs[0].unwrapped.height*32), # type: ignore
            )
        else:
            self._video_writer = None

    def get_video_writer(self, frame=None):
        if self._video_writer is None:
            assert frame is not None
            self._video_writer = cv2.VideoWriter( # type: ignore
                    self.filename,
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    self.fps,
                    (frame.shape[1], frame.shape[0]),
            )
        return self._video_writer

    def __call__(self, model, env, obs, results):
        frame = env.render()

        obs_image = draw_observations(obs)
        attn_img = draw_text('No Attention')
        rewards_img = draw_rewards(
                [
                    rew
                    for rew,reg in zip(results['reward'],results['regret'])
                    if reg is not None
                ],
                env.goal_str
        )
        frame_and_attn = concat_images(
            [
                concat_images(
                    [PIL.Image.fromarray(frame), obs_image],
                    padding=2,
                    direction='v',
                    align=-1,
                ),
                concat_images(
                    [attn_img, rewards_img],
                    padding = 5,
                    direction='h',
                )
            ],
            padding = 5,
            direction = 'v',
            align = 0,
        )

        frame_and_attn = np.array(frame_and_attn)[:,:,::-1]
        video_writer = self.get_video_writer(frame_and_attn)
        video_writer.write(frame_and_attn)

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None


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
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to test for.')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--video', type=str, default='test.webm',
                        help='Path to a video file to save.')
    parser.add_argument('--no-video', action='store_true', default=False,
                        help='Do not save a video of the episode. By default, a video is saved to "test.webm".')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to a file to save results.')
    init_parser_model(parser)
    args = parser.parse_args()

    # Create environment
    ENV_CONFIG_PRESETS = env_config_presets()
    env_config = ENV_CONFIG_PRESETS[args.env]
    #env_config['config']['num_trials'] = 5; print('--- DEBUG MODE ---')
    #env_config['config']['min_room_size'] = 4; print('--- DEBUG MODE ---')
    #env_config['config']['max_room_size'] = 4; print('--- DEBUG MODE ---')
    #env_config['config']['min_room_size'] = 6; print('--- DEBUG MODE ---')
    #env_config['config']['max_room_size'] = 6; print('--- DEBUG MODE ---')

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

    # Test model
    video_filename = os.path.abspath(args.video)
    if args.no_video:
        video_callback = None
    else:
        video_callback = VideoCallback(video_filename)

    test_results = [
            test(model, env_config, preprocess_obs, video_callback_fn=video_callback, verbose=args.verbose)
            for _ in tqdm(range(args.num_episodes))
    ]

    if video_callback is not None:
        video_callback.close()

    rewards = np.array([r['episode_reward'] for r in test_results])
    reward_mean = rewards.mean()
    reward_std = rewards.std()

    if 'step' in checkpoint:
        print(f'Checkpoint step: {checkpoint["step"]:,}')
    print(f'Rewards: {rewards.tolist()}')
    print('')

    print(f"Reward mean: {reward_mean:.2f}")
    print(f"Reward std: {reward_std:.2f}")

    if 'supervision' in test_results[0]['results']:
        supervised_rewards = np.array([r['results']['supervision']['supervised_reward'] for r in test_results])
        unsupervised_rewards = np.array([r['results']['supervision']['unsupervised_reward'] for r in test_results])
        supervised_trials = np.array([r['results']['supervision']['supervised_trials'] for r in test_results])
        unsupervised_trials = np.array([r['results']['supervision']['unsupervised_trials'] for r in test_results])

        print(f"Supervised rewards: {supervised_rewards.tolist()}")
        print(f"Unsupervised rewards: {unsupervised_rewards.tolist()}")

        print(f"Supervised reward mean / std: {supervised_rewards.mean():.2f} / {supervised_rewards.std():.2f}")
        print(f"Unsupervised reward mean / std: {unsupervised_rewards.mean():.2f} / {unsupervised_rewards.std():.2f}")

    results_by_target = [split_results_by_target(r['results']) for r in test_results]
    if max(len(r) for r in results_by_target) == 1:
        print('Single target episodes only.')
    else:
        from tabulate import tabulate
        import shutil
        tables = []
        for i,r in enumerate(results_by_target):
            non_zero = [
                (
                    target,
                    np.array(result['reward'])[np.array(result['reward']) != 0],
                )
                for target, result in r
            ]
            table_data = [
                [
                    target,
                    rewards.sum(),
                    len(rewards),
                ] for target, rewards in non_zero
            ]
            total_row = [
                'Total',
                np.sum([r.sum() for _,r in non_zero]),
                np.sum([len(r) for _,r in non_zero]),
            ]
            table = tabulate(
                table_data + [total_row],
                headers=['Target', 'Σ', '#'], tablefmt='simple_grid'
            )
            tables.append(table)
        max_table_width = max(max(len(l) for l in table.split('\n')) for table in tables)
        term_width = shutil.get_terminal_size().columns
        tables_per_row = max(1, term_width // max_table_width)
        row_width = 0
        tables_grid = [[],[]]
        for i,table in enumerate(tables):
            if len(tables_grid[-1]) >= tables_per_row:
                tables_grid.append([])
                tables_grid.append([])
            tables_grid[-1].append(table)
            tables_grid[-2].append(f'Episode {i+1}')
        table_str = tabulate(
            tables_grid,
            #headers=[f'Episode {i}' for i in range(len(tables))],
            tablefmt="plain",
        )
        print()
        print(table_str)
        print()

    print(f'Video saved to {video_filename}')

    if args.results is not None:
        results_filename = os.path.abspath(args.results)
        torch.save(test_results, results_filename)
        print(f'Results saved to {results_filename}')


    # Plot rewards as a function of trial number
    reward_by_trial = [
            r['results']['reward_by_trial']
            for r in test_results
    ]
    reward_by_trial = np.array(reward_by_trial).mean(axis=0)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    filename = os.path.abspath('reward_by_trial.png')
    plt.figure()
    plt.plot(range(len(reward_by_trial)), reward_by_trial)
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.grid()
    plt.savefig(filename)

    print(f'Plot saved to {filename}')

    breakpoint()
