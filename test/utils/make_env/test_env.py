import pytest

import gymnasium

from big_rl.utils.make_env import make_env
from pydantic import ValidationError


##################################################
# Validation
##################################################


def test_empty_config_should_error():
    config = {}
    with pytest.raises(ValueError):
        make_env(config)


def test_invalid_env_type():
    config = {
        'type': 'InvalidEnvType',
        'envs': [{
            'kwargs': {
                'id': 'FrozenLake-v1',
            },
        }],
    }
    with pytest.raises(ValueError):
        make_env(config)


def test_missing_id():
    config = {
        'type': 'Env',
        'envs': [{
            'kwargs': {},
        }],
    }
    with pytest.raises(KeyError):
        make_env(config)


def test_extra_params():
    config = {
        'type': 'Env',
        'envs': [{
            'kwargs': {
                'id': 'FrozenLake-v1',
            },
        }],
        'extra_params': None,
    }
    with pytest.raises(ValidationError):
        make_env(config)


##################################################
# Single environment
##################################################


def test_frozenlake():
    config = {
        'type': 'Env',
        'envs': [{
            'kwargs': {
                'id': 'FrozenLake-v1',
            },
        }],
    }

    # Initialize without error
    env = make_env(config)
    assert isinstance(env, gymnasium.Env)

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)


def test_atari():
    config = {
        'type': 'Env',
        'envs': [{
            'kwargs': {
                'id': 'ALE/Pong-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
            }
        }],
    }

    # Initialize without error
    env = make_env(config)
    assert isinstance(env, gymnasium.Env)

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)


def test_atari_with_wrappers():
    config = {
        'type': 'Env',
        'envs': [{
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
        }],
    }

    # Initialize without error
    env = make_env(config)
    assert isinstance(env, gymnasium.Env)

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

        assert obs.shape == (29, 29)
