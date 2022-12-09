import argparse
import itertools
import sys

import torch
import numpy as np
import pygame

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model, merge_space


def test(model, env_config, preprocess_obs_fn):
    env = make_env(**env_config)

    # Get a sample frame from the environment so we know how big the image is
    env.reset()
    frame = env.render()
    assert frame is not None
    pygame.init()
    screen = pygame.display.set_mode(frame.shape[:2])

    #while True:
    #    for event in pygame.event.get():
    #        if event.type == pygame.QUIT:
    #            sys.exit()

    #    screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
    #    pygame.display.flip()

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

    obs, info = env.reset()
    obs = preprocess_obs_fn(obs)

    fps = 30
    user_reward = 0
    for _ in itertools.count():
        pygame.time.wait(int(1000/fps))

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    sys.exit()
                if event.key == pygame.K_r:
                    hidden = model.init_hidden(1)
                    obs, info = env.reset()
                    obs = preprocess_obs_fn(obs)
                if event.key == pygame.K_GREATER:
                    fps += 1
                if event.key == pygame.K_LESS:
                    fps = max(1, fps-1)
                if event.key == pygame.K_KP_PLUS:
                    user_reward = 1
                    print('Reward: ', user_reward)
                if event.key == pygame.K_KP_MINUS:
                    user_reward = -1
                    print('Reward: ', user_reward)

        # Run a step in the environment
        with torch.no_grad():
            model_output = model(obs, hidden)

            hidden = model_output['hidden']

            action_probs = model_output['action'].softmax(1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().cpu().numpy().squeeze()

        obs, reward, _, _, info = env.step(action)
        obs['obs (shaped_reward)'] = np.array([user_reward], dtype=np.float32)
        obs = preprocess_obs_fn(obs)

        # Record results
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

        # Render the environment
        frame = env.render()

        screen.blit(pygame.surfarray.make_surface(frame), (0, 0))

        rect = pygame.Surface((120,40))
        rect.set_alpha(128)
        rect.fill((0,0,0))
        screen.blit(rect, (0,0))

        font = pygame.font.SysFont('Arial', 20)

        screen.blit(font.render(f'FPS: {fps}', True, (255,255,255)), (0, 0))
        screen.blit(font.render(f'Reward: {user_reward}', True, (255,255,255)), (0, 20))
        pygame.display.flip()

        # Reset user input
        user_reward = 0

    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'results': results,
    }


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
    parser.add_argument('--results', type=str, default=None,
                        help='Path to a file to save results.')

    init_parser_model(parser)

    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode')

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
            device = device,
    )
    model.to(device)

    if args.model is not None:
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f'Loaded checkpoint from {args.model}')
    else:
        if not args.debug:
            raise ValueError('No model specified')
        else:
            print('No model specified, using random model. (debug mode)')

    # Test model
    test(model, env_config, preprocess_obs)
