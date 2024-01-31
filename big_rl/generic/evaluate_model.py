import argparse
from collections import defaultdict
from enum import Enum
import itertools
import os
from typing import Tuple, Optional
import yaml

import cv2
import numpy as np
import torch
from tqdm import tqdm
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
from fonts.ttf import Roboto # type: ignore
from big_rl.model.modular_policy_8 import ModularPolicy8
from big_rl.model.recurrent_attention_16 import RecurrentAttention16 # type: ignore

from big_rl.mujoco.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.mujoco.common import env_config_presets, init_model
from big_rl.utils import merge_space
from big_rl.model import factory as model_factory
from big_rl.utils.make_env import EnvGroup, get_config_from_yaml, make_env_from_yaml
from big_rl.generic.script import get_action_dist_function


ALIGN_LEFT = -1
ALIGN_RIGHT = 1
ALIGN_TOP = -1
ALIGN_BOTTOM = 1
ALIGN_CENTER = 0


def test(model, env_config, preprocess_obs_fn, video_callback_fn={}, verbose=False, render=False, num_episodes=1, warmup_episodes=0):
    if warmup_episodes > 0:
        envs = make_env_from_yaml(env_config)
        output = defaultdict(list)
        for env in envs:
            for _ in tqdm(range(warmup_episodes), desc='Warmup'):
                test2(model, env, preprocess_obs_fn, video_callback_fn, verbose)
            for _ in tqdm(range(num_episodes), desc='Test'):
                output[env.name].append(test2(
                    model, env, preprocess_obs_fn, video_callback_fn.get(env.name), verbose))
        return output
    else:
        # If `warmup_episodes` is 0, we will recreate the environment for each episode to ensure they all start from the same state.
        output = defaultdict(list)
        for ep in tqdm(range(num_episodes), desc='Test'):
            envs = make_env_from_yaml(env_config)
            for env in envs:
                output[env.name].append(test2(
                    model,
                    env.env,
                    preprocess_obs_fn,
                    video_callback_fn.get(env.name, None),
                    verbose=verbose,
                    render=render,
                    metadata={'episode': ep, 'task_name': env.name},
                ))
        return output
        #return [
        #    test2(
        #        model,
        #        make_env_from_yaml(env_config)[0].env,
        #        preprocess_obs_fn,
        #        video_callback_fn,
        #        verbose=verbose,
        #        render=render,
        #        metadata={'episode': ep},
        #    )
        #    for ep in tqdm(range(num_episodes), desc='Test')
        #]


def test2(model, env, preprocess_obs_fn, video_callback_fn=None, verbose=False, render=False, metadata={}):
    action_dist_fn = get_action_dist_function(env.single_action_space)

    episode_reward = 0 # total reward for the current episode
    episode_length = 0 # Number of transitions in the episode
    results = {
            'reward': [],
            'regret': [],
            'attention': [],
            'hidden': [],
            'input_labels': [],
            'target': [], # Goal string
            'shaped_reward': [], # Predicted shaped reward
    }

    hidden = model.init_hidden(1) # type: ignore (???)
    steps_iterator = itertools.count()
    #steps_iterator = range(100+np.random.randint(50)); print('--- DEBUG MODE ---')
    #steps_iterator = range(10); print('--- DEBUG MODE ---')
    #if verbose:
    steps_iterator = tqdm(steps_iterator)

    obs, info = env.reset()
    obs = preprocess_obs_fn(obs)
    terminated = False
    truncated = False
    for step in steps_iterator:
        with torch.no_grad():
            model_output = model(obs, hidden)

            hidden = model_output['hidden']

            action_dist, _ = action_dist_fn(model_output)
            action = action_dist.sample().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        obs = preprocess_obs_fn(obs)
        if render:
            env.render()

        results['reward'].append(reward)
        results['regret'].append(info.get('regret', None))
        #results['model_output'] = model_output
        #if model.has_attention:
        #    results['attention'].append((
        #        [x.numpy() for x in model.last_attention],
        #        [x.numpy() for x in model.last_ff_gating],
        #        {k: v.numpy() for k,v in model.last_output_attention.items()},
        #    ))
        #results['hidden'].append([
        #    x.cpu().detach().numpy() for x in model.last_hidden
        #])
        #results['input_labels'].append(model.last_input_labels)
        #results['target'] = env.goal_str # type: ignore
        #results['target'] = 'locomotion and stuff'
        #results['shaped_reward'].append(compute_shaped_reward(model))

        episode_reward += info.get('reward', reward)
        episode_length += 1

        if verbose:
            if reward != 0:
                tqdm.write(f"Step {episode_length}: reward={reward} total={episode_reward}")

        if terminated or truncated:
            tqdm.write(f"Step {episode_length}: total reward={episode_reward}")
            break

        if video_callback_fn is not None:
            video_callback_fn(model, model_output, env.envs[0], obs, results, {**metadata, 'step': step})

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        #'episode_reward': info['episode']['r'],
        #'episode_length': info['episode']['l'],
        'results': results,
    }



def concat_images(images: list[PIL.Image.Image], padding=0, direction='h', align=0) -> PIL.Image.Image:
    """
    Concatenate images along a given direction.

    Args:
        images: list of PIL images or callable functions that return PIL images. The functions must take two arguments: width and height.
    """
    if len(images) == 0:
        return PIL.Image.new('RGB', (0, 0))
    if len(images) == 1:
        return images[0]

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

def draw_grid(data, xlabels=None, ylabels=None):
    block_size = 24
    padding = 2
    font_size = 18

    font_family = Roboto
    font = PIL.ImageFont.truetype(font_family, font_size)

    all_images = []

    # Labels
    if xlabels is not None:
        xlabels_img_list = []
        for label in xlabels:
            _, _, text_width, text_height = font.getbbox(label)
            width = text_width
            height = block_size

            img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
            draw = PIL.ImageDraw.Draw(img)
            draw.fontmode = 'L' # type: ignore
            draw.text(
                    (0,0),
                    label,
                    font=font,
                    fill=(0,0,0)
            )
            xlabels_img_list.append(img)
        xlabels_img = concat_images(xlabels_img_list, padding=padding, direction='v', align=-1).rotate(90, expand=True)
        all_images.append(xlabels_img)

    # Grid
    width = data.shape[1]*block_size + (data.shape[1]+1)*padding
    height = data.shape[0]*block_size + (data.shape[0]+1)*padding
    img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            weight = data[i,j].item()
            c = int(255*(1-weight))
            x = j*(block_size+padding) + padding
            y = i*(block_size+padding) + padding
            PIL.ImageDraw.Draw(img).rectangle(
                    (x,y,x+block_size,y+block_size),
                    fill=(c,c,c),
            )
    all_images.append(img)

    return concat_images(all_images, padding=padding, direction='v', align=ALIGN_LEFT)

def draw_text(text, font_family=Roboto, font_size=18, color=(0,0,0), padding=2, height=None):
    font = PIL.ImageFont.truetype(font_family, font_size)

    _, _, text_width, text_height = font.getbbox(text)
    img = PIL.Image.new('RGB',
            (text_width+2*padding, height or text_height+2*padding),
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


def draw_attention_mm1(model_output):
    block_size = 24
    padding = 2
    font_size = 18

    output_attention = {k:x['attn_output_weights'] for k,x in model_output['misc']['output'].items()}
    input_labels = model_output['misc']['input']['input_labels']
    core_labels = model_output['misc']['core'].get('output_labels', [])

    # Core modules
    def draw_core_attention(data) -> PIL.Image.Image:
        if 'container_type' in data:
            attn_images_list = []
            for i in itertools.count():
                if i not in data:
                    break
                attn_images_list.append(draw_core_attention(data[i]))
            direction = 'v' if data['container_type'] == 'series' else 'h'
            attn_images = concat_images(attn_images_list, padding=padding, direction=direction, align=ALIGN_CENTER)
            return attn_images
        else:
            images = []

            core_attention = data.get('attn_output_weights')
            if core_attention is None:
                return draw_text('N/A')
            core_images_concat = draw_grid(core_attention[:,0,:], xlabels=input_labels+core_labels)
            images.append(core_images_concat)

            if 'gates' in data:
                gates = data['gates']
                gate_images = []
                for label,attn in gates.items():
                    gate_images = []
                    gate_images.append(draw_grid(attn[:,0,:].mean(dim=1,keepdims=True)))
                gate_images_concat = concat_images(gate_images, padding=padding, direction='h', align=ALIGN_CENTER)
                images.append(gate_images_concat)

            return concat_images(images, padding=padding, direction='h', align=ALIGN_BOTTOM)
    core_images_concat = draw_core_attention(model_output['misc']['core'])

    # Output modules
    output_label_images = []
    output_attn_images = []
    for k,v in output_attention.items():
        output_label_images.append(draw_text(k, font_size=font_size, height=block_size+2*padding))
        output_attn_images.append(draw_grid(v[:,0,:]))
    output_images_concat = concat_images([
        concat_images(
            output_label_images, padding=padding, direction='v', align=ALIGN_RIGHT),
        concat_images(
            output_attn_images, padding=padding, direction='v', align=ALIGN_LEFT),
    ], padding=padding, direction='h', align=ALIGN_CENTER)

    all_images_concat = concat_images([
        core_images_concat,
        output_images_concat,
    ], padding=padding, direction='v', align=ALIGN_CENTER)

    return all_images_concat


class VideoCallback:
    def __init__(self, filename: str, 
                 size: Optional[Tuple[int,int]] = None,
                 fps: int = 30):
        if not filename.endswith('.webm'):
            raise ValueError('filename must end with .webm')
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
            self._frame_size = (frame.shape[1], frame.shape[0])
            self._video_writer = cv2.VideoWriter( # type: ignore
                    self.filename,
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    self.fps,
                    self._frame_size,
            )
        return self._video_writer

    def __call__(self, model, model_output, env, obs, results, metadata={}):
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        attn_img = draw_attention_mm1(model_output)

        data_img = concat_images([
            draw_text(f'Task: {metadata["task_name"]}'),
            draw_text(f'Episode: {metadata["episode"]}'),
            draw_text(f'Step: {len(results["reward"])}'),
            draw_text(f'Total Reward: {np.sum(results["reward"]):.2f}'),
            draw_text(f'Energy Remaining: {obs.get("obs (energy)", torch.tensor(float("inf"))).item():.2f}'),
        ], direction='v', padding=2, align=ALIGN_LEFT)

        frame_and_attn = concat_images(
            [
                PIL.Image.fromarray(frame),
                concat_images([
                    data_img,
                    attn_img,
                ], direction='v', align=ALIGN_LEFT),
            ],
            padding = 5,
            direction = 'h',
            align = ALIGN_TOP,
        )

        final_image = np.array(frame_and_attn)
        video_writer = self.get_video_writer(final_image)

        # Resize frame if it's not the right size
        if final_image.shape[:2] != self._frame_size:
            final_image = PIL.Image.new('RGB', self._frame_size, color=(255,255,255))
            final_image.paste(frame_and_attn)
            final_image = np.array(final_image)

        video_writer.write(final_image)

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None


def init_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-config', type=str, default=None,
                        help='Path to an environment config file (yaml format). If specified, all other environment arguments are ignored.')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model checkpoint to test.')
    parser.add_argument('--model-envs-config', type=str, nargs='*', default=None,
                        help='Environments whose observation spaces are used to initialize the model in case they are different from the specs of the test environment. If not specified, the "--env-config" test environment will be used.')
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to test for.')
    parser.add_argument('--warmup-episodes', type=int, default=0,
                        help='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment during testing. The episode cannot be saved to video if this is enabled.')
    parser.add_argument('--video', type=str, default='test.webm',
                        help='Path to a video file to save.')
    parser.add_argument('--no-video', action='store_true', default=False,
                        help='Do not save a video of the episode. By default, a video is saved to "test.webm".')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to a file to save results.')
    init_parser_model(parser)
    return parser


def main(args):
    # Create environment
    if args.env_config is not None:
        envs = make_env_from_yaml(args.env_config)
    else:
        raise Exception('Environments must be configured via a config file.')

    # Validate environment
    for env in envs:
        if env.env.num_envs != 1:
            raise Exception('Test environments must have 1 environment per process.')

    # Load model
    device = torch.device('cpu')
    if args.model_config is not None:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        observation_spaces = defaultdict(lambda: [])
        action_spaces = defaultdict(lambda: [])
        for env in envs:
            if env.model_name is None:
                continue
            observation_spaces[env.model_name].append(env.env.single_observation_space)
            action_spaces[env.model_name].append(env.env.single_action_space)
        observation_spaces = {k: merge_space(*v) for k,v in observation_spaces.items()}
        action_spaces = {k: merge_space(*v) for k,v in action_spaces.items()}

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

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f'Loaded checkpoint from {args.model}')
    else:
        print('No model checkpoint specified, using random initialization.')
        checkpoint = {}

    if 'step' in checkpoint:
        print(f'Checkpoint step: {checkpoint["step"]:,}')

    # Test model
    video_filenames = {
        env.name: os.path.abspath(args.video).format(name=env.name)
        for env in envs
    }
    if args.render or args.no_video:
        video_callback = {}
    else:
        video_callback = {
            env.name: VideoCallback(video_filenames[env.name])
            for env in envs
        }
        if len(video_callback) != len(set([env.name for env in envs])):
            raise Exception(f'Task names must be unique: {[env.name for env in envs]}')
        if len(video_callback) != len(set(video_filenames.values())):
            raise Exception(f'Multiple tasks are assigned the same video output. Make sure to set the `--video` parameter to include "{{name}}" for the task name.')

    #obs_scale = {'obs': 1.0/255.0}
    obs_scale = {}
    obs_ignore = ['obs (mission)']
    def preprocess_obs_fn(obs):
        return {
            k: torch.tensor(v, dtype=torch.float, device=device)*obs_scale.get(k,1)
            for k,v in obs.items() if k not in obs_ignore
        }

    test_results = test(model, args.env_config, preprocess_obs_fn, video_callback_fn=video_callback, verbose=args.verbose, num_episodes=args.num_episodes, warmup_episodes=args.warmup_episodes)

    if video_callback is not None:
        for vc in video_callback.values():
            vc.close()

    rewards = {
        k: np.array([r['episode_reward'] for r in v])
        for k,v in test_results.items()
    }
    reward_mean = {k:v.mean() for k,v in rewards.items()}
    reward_std = {k:v.std() for k,v in rewards.items()}

    for k in reward_mean.keys():
        print('-'*80)
        print(f'Task: {k}')
        print('-'*80)
        if 'step' in checkpoint:
            print(f'Checkpoint step: {checkpoint["step"]:,}')
        print(f'Rewards: {rewards[k].tolist()}')
        print('')

        print(f"Reward mean: {reward_mean[k]:.2f}")
        print(f"Reward std: {reward_std[k]:.2f}")
        if k in video_callback:
            print(f'Video saved to {video_filenames[k]}')

    print('-'*80)
    print('sftp:')
    if len(video_callback) > 0:
        for k in reward_mean.keys():
            print(f'get {video_filenames[k]}')

    print('-'*80)
    if args.results is not None:
        results_filename = os.path.abspath(args.results)
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        torch.save(test_results, results_filename)
        print(f'Results saved to {results_filename}')


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)

