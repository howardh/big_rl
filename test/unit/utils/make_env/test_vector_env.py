import gymnasium

from big_rl.utils.make_env import make_env


def test_atari_one_type():
    config = {
        'type': 'SyncVectorEnv',
        'envs': [{
            'repeat': 5,
            'kwargs': {
                'id': 'ALE/Pong-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
            }
        }],
    }

    # Initialize without error
    env = make_env([config])[0]

    assert isinstance(env.env, gymnasium.vector.SyncVectorEnv)
    assert env.env.num_envs == 5
    assert len(env.env_labels) == 5

    # Reset and run a few steps without error
    env.env.reset()
    for _ in range(5):
        action = env.env.action_space.sample()
        env.env.step(action)


def test_atari_two_types():
    """ Two different types of environments in one vector env """
    config = {
        'type': 'SyncVectorEnv',
        'envs': [{
            'repeat': 5,
            'kwargs': {
                'id': 'ALE/Pong-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
                'full_action_space': True,
            }
        }, {
            'repeat': 3,
            'kwargs': {
                'id': 'ALE/Breakout-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
                'full_action_space': True,
            }
        }],
    }

    # Initialize without error
    env_group = make_env([config])[0]
    env = env_group.env

    assert isinstance(env_group.env, gymnasium.vector.SyncVectorEnv)
    assert env_group.env.num_envs == 5 + 3
    assert len(env_group.env_labels) == 5 + 3

    # Three of them should be Breakout, and five should be Pong
    assert env_group.env.envs[0].unwrapped.spec.id == 'ALE/Pong-v5'  # type: ignore
    assert env_group.env.envs[1].unwrapped.spec.id == 'ALE/Pong-v5'  # type: ignore
    assert env_group.env.envs[2].unwrapped.spec.id == 'ALE/Pong-v5'  # type: ignore
    assert env_group.env.envs[3].unwrapped.spec.id == 'ALE/Pong-v5'  # type: ignore
    assert env_group.env.envs[4].unwrapped.spec.id == 'ALE/Pong-v5'  # type: ignore
    assert env_group.env.envs[5].unwrapped.spec.id == 'ALE/Breakout-v5'  # type: ignore
    assert env_group.env.envs[6].unwrapped.spec.id == 'ALE/Breakout-v5'  # type: ignore
    assert env_group.env.envs[7].unwrapped.spec.id == 'ALE/Breakout-v5'  # type: ignore

    # Reset and run a few steps without error
    env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        env.step(action)


def test_env_labels_default():
    """  """
    config = [{
        'type': 'SyncVectorEnv',
        'envs': [{
            'repeat': 5,
            'kwargs': {
                'id': 'ALE/Pong-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
                'full_action_space': True,
            }
        }, {
            'repeat': 3,
            'kwargs': {
                'id': 'ALE/Breakout-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
                'full_action_space': True,
            }
        }],
    }]

    labels = make_env(config)[0].env_labels
    assert labels == [
            'ALE/Pong-v5',
            'ALE/Pong-v5',
            'ALE/Pong-v5',
            'ALE/Pong-v5',
            'ALE/Pong-v5',
            'ALE/Breakout-v5',
            'ALE/Breakout-v5',
            'ALE/Breakout-v5',
    ]


def test_env_labels_custom_names():
    """  """
    config = {
        'type': 'SyncVectorEnv',
        'envs': [{
            'name': 'doot',
            'repeat': 5,
            'kwargs': {
                'id': 'ALE/Pong-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
                'full_action_space': True,
            }
        }, {
            'name': 'boop',
            'repeat': 3,
            'kwargs': {
                'id': 'ALE/Breakout-v5',
                'render_mode': 'rgb_array',
                'frameskip': 1,
                'full_action_space': True,
            }
        }],
    }

    labels = make_env([config])[0].env_labels
    assert labels == [
            'doot',
            'doot',
            'doot',
            'doot',
            'doot',
            'boop',
            'boop',
            'boop',
    ]
