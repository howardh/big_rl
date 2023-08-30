import pytest

import gymnasium

from big_rl.utils.make_env import make_single_env
from pydantic import ValidationError


##################################################
# Validation
##################################################


def test_empty_config_should_error():
    config = {}
    with pytest.raises(ValueError):
        make_single_env(config)


##################################################
# Environment Creation
##################################################

def test_frozenlake():
    config = {
        'kwargs': {
            'id': 'FrozenLake-v1',
        },
    }

    # Initialize without error
    env = make_single_env(config)
    assert isinstance(env, gymnasium.Env)

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)


def test_atari():
    config = {
        'kwargs': {
            'id': 'ALE/Pong-v5',
            'render_mode': 'rgb_array',
            'frameskip': 1,
        }
    }

    # Initialize without error
    env = make_single_env(config)
    assert isinstance(env, gymnasium.Env)

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)


def test_atari_with_wrappers():
    config = {
        'kwargs': {
            'id': 'ALE/Pong-v5',
            'render_mode': 'rgb_array',
            'frameskip': 1,
        },
        'wrappers': [{
            'type': 'AtariPreprocessing',
            'kwargs': {
                'screen_size': 29,
            },
        }]
    }

    # Initialize without error
    env = make_single_env(config)
    assert isinstance(env, gymnasium.Env)

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        assert obs.shape == (29, 29)

