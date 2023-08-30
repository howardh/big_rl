import argparse
import itertools
import os
from typing import Tuple, Optional, Any
import yaml

import cv2
import numpy as np
import torch
from tqdm import tqdm
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
from fonts.ttf import Roboto # type: ignore

from big_rl.atari.script import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.atari.common import env_config_presets, init_model
from big_rl.utils import merge_space
from big_rl.model import factory as model_factory


def test(model, env_config, preprocess_obs_fn, video_callback_fn=None, hidden_info_fn=None, verbose=False):
    env = make_env(**env_config)

    episode_reward = 0 # total reward for the current episode
    episode_length = 0 # Number of transitions in the episode
    results: dict[str,Any] = {
            'reward': [],
            'attention': [],
            'hidden': [],
            'input_labels': [],
            'hidden_info': None, # Last output of hidden_info_fn. No history is kept.
            'foo': [] # Output attention
    }

    hidden = model.init_hidden(1) # type: ignore (???)
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
            model_output = model(obs, hidden)

            hidden = model_output['hidden']

            action_probs = model_output['action'].softmax(1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy().squeeze()

        obs, reward, terminated, truncated, info = env.step(action)
        obs = preprocess_obs_fn(obs)

        results['foo'].append(model_output['misc']['output']['value']['attn_output_weights'].detach().cpu().numpy())
        results['reward'].append(reward)
        #results['hidden'].append([
        #    x.cpu().detach().numpy() for x in model.last_hidden
        #])
        #results['input_labels'].append(model.last_input_labels)

        episode_reward += float(reward)
        episode_length += 1

        if verbose:
            if reward != 0:
                tqdm.write(f"Step {episode_length}: reward={reward} total={episode_reward}")

        if terminated or truncated:
            break

        if video_callback_fn is not None:
            video_callback_fn(model, env, obs, results)

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'results': results,
    }


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
            self._video_writer = cv2.VideoWriter( # type: ignore
                    self.filename,
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    self.fps,
                    (frame.shape[1], frame.shape[0]),
            )
        return self._video_writer

    def __call__(self, model, env, obs, results):
        frame = env.render()

        #obs_image = draw_observations(obs)
        #if model.has_attention:
        #    attn_img = draw_attention(
        #            core_attention = model.last_attention,
        #            query_gating = model.last_ff_gating,
        #            output_attention = model.last_output_attention,
        #            input_labels = results['input_labels'][-1],
        #    )
        #else:
        #    attn_img = draw_text('No Attention')
        #rewards_img = draw_rewards(
        #        [
        #            rew
        #            for rew,reg in zip(results['reward'],results['regret'])
        #            if reg is not None
        #        ],
        #        env.goal_str
        #)
        #hidden_info_img = draw_hidden_info(results['hidden_info'])
        #frame_and_attn = concat_images(
        #    [
        #        concat_images(
        #            [PIL.Image.fromarray(frame), obs_image],
        #            padding=2,
        #            direction='v',
        #            align=-1,
        #        ),
        #        concat_images(
        #            [attn_img, rewards_img],
        #            padding = 5,
        #            direction='h',
        #        )
        #    ],
        #    padding = 5,
        #    direction = 'v',
        #    align = 0,
        #)
        #final_img = concat_images(
        #        [frame_and_attn, hidden_info_img],
        #        padding = 5,
        #        direction='h',
        #        align=0,
        #)
        final_img = PIL.Image.fromarray(frame)

        final_img = np.array(final_img)[:,:,::-1]
        video_writer = self.get_video_writer(final_img)
        video_writer.write(final_img)

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None


def preprocess_obs(obs):
    obs_scale = { 'obs': 1.0 / 255.0 }
    obs_ignore = []
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
    parser.add_argument('--env-mode', type=int, default=None,
                        help='Override game mode of the evaluation environment.')
    parser.add_argument('--env-difficulty', type=int, default=None,
                        help='Override game difficulty of the evaluation environment.')
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

    if args.env_mode is not None:
        env_config['config']['mode'] = args.env_mode
    if args.env_difficulty is not None:
        env_config['config']['difficulty'] = args.env_difficulty
    env = make_env(**env_config)
    if args.model_envs is not None:
        dummy_envs = [
                make_env(**ENV_CONFIG_PRESETS[c]) for c in args.model_envs]
    else:
        dummy_envs = [env]
    observation_space = merge_space(*[e.observation_space for e in dummy_envs])

    # Load model
    device = torch.device('cpu')
    if args.model_config is not None:
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        model = model_factory.create_model(
            model_config,
            observation_space = observation_space,
            action_space = env.action_space,
        )
    else:
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
        model.load_state_dict(checkpoint['model'], strict=True)
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

    print(f'Video saved to {video_filename}')

    if args.results is not None:
        results_filename = os.path.abspath(args.results)
        torch.save(test_results, results_filename)
        print(f'Results saved to {results_filename}')

    ## Temporary plotting code. Remove when done.
    #from matplotlib import pyplot as plt
    #output_attn_weights = np.concatenate([np.concatenate(r['results']['foo']).squeeze() for r in test_results])
    #plt.figure()
    #labels = ['Input #1', 'Input #2', 'Input #3', 'Old Core #1', 'Old Core #2', 'Old Core #3', 'New Core']
    #plt.boxplot(output_attn_weights, labels=labels)
    #plt.xticks(rotation=30, ha='center')
    #plt.ylabel('Attention Weight')
    #plt.savefig('output_attn_weights.png')

    breakpoint()
