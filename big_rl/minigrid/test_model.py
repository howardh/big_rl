import argparse
import itertools
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
from fonts.ttf import Roboto # type: ignore

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model


def test(model, env_config, preprocess_obs_fn, video_callback_fn=None, verbose=False):
    env = make_env(**env_config)

    episode_reward = 0 # total reward for the current episode
    episode_length = 0 # Number of transitions in the episode
    results = {
            'reward': [],
            'regret': [],
            'attention': [],
            'hidden': [],
            'input_labels': [],
            'target': [], # Goal string
    }

    hidden = model.init_hidden(1) # type: ignore (???)
    steps_iterator = itertools.count()
    if verbose:
        steps_iterator = tqdm(steps_iterator)

    obs, info = env.reset()
    obs = preprocess_obs_fn(obs)
    terminated = False
    truncated = False
    for _ in steps_iterator:
        with torch.no_grad():
            model_output = model(obs, hidden)

            hidden = model_output['hidden']

            action_probs = model_output['action'].softmax(1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(action)
        obs = preprocess_obs_fn(obs)

        results['reward'].append(reward)
        results['regret'].append(info.get('regret', None))
        results['attention'].append((
            [x.numpy() for x in model.last_attention],
            [x.numpy() for x in model.last_ff_gating],
            {k: v.numpy() for k,v in model.last_output_attention.items()},
        ))
        results['hidden'].append([
            x.cpu().detach().numpy() for x in model.last_hidden
        ])
        results['input_labels'].append(model.last_input_labels)
        results['target'] = env.goal_str # type: ignore

        episode_reward += reward
        episode_length += 1

        if verbose:
            if reward != 0:
                tqdm.write(f"Step {episode_length}: reward={reward} total={episode_reward}")

        if terminated or truncated:
            break

        video_callback_fn(model, env, results)

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
    }


def concat_images(images, padding=0, direction='h', align=0):
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

def draw_attention(core_attention, query_gating, output_attention, input_labels):
    block_size = 24
    padding = 2

    font_family = Roboto
    font_size = 18
    font = PIL.ImageFont.truetype(font_family, font_size)

    # Core modules
    core_labels = []
    for label in input_labels:
        text_width, text_height = font.getsize(label)
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
        core_labels.append(img)
    core_labels_concat = concat_images(core_labels, padding=padding, direction='v', align=-1)
    core_labels_concat = core_labels_concat.rotate(90, expand=True)

    core_images = []
    for layer in core_attention:
        num_blocks, _, num_inputs = layer.shape
        width = num_inputs*block_size + (num_inputs+1)*padding
        height = num_blocks*block_size + (num_blocks+1)*padding
        img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
        for i in range(num_blocks):
            for j in range(num_inputs):
                weight = layer[i,0,j].item()
                c = int(255*(1-weight))
                x = j*(block_size+padding) + padding
                y = i*(block_size+padding) + padding
                PIL.ImageDraw.Draw(img).rectangle(
                        (x,y,x+block_size,y+block_size),
                        fill=(c,c,c),
                )
        core_images.append(img)

    core_images[0] = concat_images([core_labels_concat, core_images[0]], padding=padding, direction='v', align=-1) # Attach labels to first layer attention
    core_images_concat = concat_images(core_images, padding=padding, direction='v', align=1)

    # Gating
    num_layers = len(query_gating)
    max_layer_size = max(layer.shape[0] for layer in query_gating)
    width = num_layers*block_size + (num_layers+1)*padding
    height = max_layer_size*block_size + (max_layer_size+1)*padding
    query_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
    for i, layer in enumerate(query_gating):
        num_blocks = layer.shape[0]
        for j in range(num_blocks):
            weight = layer[j,0].item()
            c = int(255*(1-weight))
            x = i*(block_size+padding) + padding
            y = j*(block_size+padding) + padding
            PIL.ImageDraw.Draw(query_image).rectangle(
                    (x,y,x+block_size,y+block_size),
                    fill=(c,c,c)
            )

    # Output modules
    output_images = {}
    for k, layer in output_attention.items():
        layer = layer.squeeze()
        num_inputs = len(layer)
        width = num_inputs*block_size + (num_inputs+1)*padding
        height = block_size + 2*padding
        img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
        for i in range(num_inputs):
            weight = layer[i].item()
            c = int(255*(1-weight))
            x = i*(block_size+padding) + padding
            y = padding
            PIL.ImageDraw.Draw(img).rectangle(
                    (x,y,x+block_size,y+block_size),
                    fill=(c,c,c)
            )
        output_images[k] = img

    text_images = {}
    for k in output_attention.keys():
        text_width, text_height = font.getsize(k)
        img = PIL.Image.new('RGB',
                (text_width+2*padding, text_height+2*padding),
                color=(255,255,255))
        draw = PIL.ImageDraw.Draw(img)
        draw.fontmode = 'L' # type: ignore
        draw.text(
                (padding, padding),
                k,
                font=font,
                fill=(0,0,0)
        )
        text_images[k] = img

    output_images_concat = concat_images(
            [
                concat_images(
                    [
                        text_images[k],
                        output_images[k],
                    ],
                    padding = padding,
                    direction='v',
                    align=-1,
                )
                for k in output_images.keys()],
            padding=padding, direction='v'
    )

    all_images_concat = concat_images(
            [
                core_images_concat,
                query_image,
                output_images_concat,
            ],
            padding=padding, direction='h'
    )

    return all_images_concat

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


class VideoCallback:
    def __init__(self, filename: str, 
                 size: Tuple[int,int] = None,
                 fps: int = 30):
        assert filename.endswith('.webm')

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
            self._video_writer = cv2.VideoWriter( # type: ignore
                    self.filename,
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    self.fps,
                    (frame.shape[1], frame.shape[0]),
            )
        return self._video_writer

    def __call__(self, model, env, results):
        frame = env.render()

        attn_img = draw_attention(
                core_attention = model.last_attention,
                query_gating = model.last_ff_gating,
                output_attention = model.last_output_attention,
                input_labels = results['input_labels'][-1],
        )
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
                PIL.Image.fromarray(frame),
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
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to test for.')
    parser.add_argument('--verbose', action='store_true', default=False)
    init_parser_model(parser)
    args = parser.parse_args()

    # Create environment
    ENV_CONFIG_PRESETS = env_config_presets()
    env_config = ENV_CONFIG_PRESETS[args.env]

    env = make_env(**env_config)

    # Load model
    device = torch.device('cpu')
    model = init_model(
            observation_space = env.observation_space,
            action_space = env.action_space,
            model_type = args.model_type,
            recurrence_type = args.recurrence_type,
            architecture = args.architecture,
            device = device,
    )
    model.to(device)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded checkpoint from {args.model}')

    # Test model
    video_filename = os.path.abspath('test.webm')
    video_callback = VideoCallback(video_filename)

    test_results = [
            test(model, env_config, preprocess_obs, video_callback_fn=video_callback, verbose=args.verbose)
            for _ in tqdm(range(args.num_episodes))
    ]

    video_callback.close()

    rewards = np.array([r['episode_reward'] for r in test_results])
    reward_mean = rewards.mean()
    reward_std = rewards.std()

    print(f'Rewards: {rewards.tolist()}')
    print('')

    print(f"Reward mean: {reward_mean:.2f}")
    print(f"Reward std: {reward_std:.2f}")
    print(f'Video saved to {video_filename}')
