import pytest
import itertools

import numpy as np

from big_rl.minigrid.envs import MultiRoomEnv_v1 as Env


# https://github.com/Farama-Foundation/Minigrid/blob/b7d447b34fc05565b05a28304e5b981a0a714b05/minigrid/minigrid_env.py#L33
TURN_LEFT = 0
TURN_RIGHT = 1
MOVE_FORWARD = 2
PICKUP = 3
ACTION_CYCLE = [PICKUP, TURN_LEFT]

ENV_CONFIG = {
    'min_num_rooms': 1,
    'max_num_rooms': 1,
    'min_room_size': 4,
    'max_room_size': 4,
    'num_trials': 5,
    'fetch_config': {
        'num_objs': 3,
    },
    'shaped_reward_config': {
        'type': 'subtask',
    },
}


def test_no_transform():
    env = Env(**ENV_CONFIG)

    env.reset()
    for i in itertools.count():
        obs, reward, terminated, truncated, _ = env.step(ACTION_CYCLE[i % len(ACTION_CYCLE)])

        assert reward == obs['shaped_reward']

        if terminated or truncated:
            break


def test_1_step_delay():
    config = {
        **ENV_CONFIG,
        'shaped_reward_config': {
            'type': 'subtask',
            'delay': ('fixed', 1),
        },
    }
    env = Env(**config)

    prev_reward = 0

    env.reset()
    for i in itertools.count():
        obs, reward, terminated, truncated, _ = env.step(ACTION_CYCLE[i % len(ACTION_CYCLE)])

        assert prev_reward == obs['shaped_reward']
        prev_reward = reward

        if terminated or truncated:
            break


@pytest.mark.parametrize('n', [0, 1, 2])
def test_n_trial_delayed_start(n):
    config = {
        **ENV_CONFIG,
        'shaped_reward_config': {
            'type': 'subtask',
            'delayed_start': ('fixed', n, 'trials'),
        },
    }
    env = Env(**config)

    rewards = []
    shaped_rewards = []

    env.reset()
    for i in itertools.count():
        obs, reward, terminated, truncated, _ = env.step(ACTION_CYCLE[i % len(ACTION_CYCLE)])

        rewards.append(reward)
        shaped_rewards.append(obs['shaped_reward'].item())

        if terminated or truncated:
            break

    # Skip the first `n` rewards. Everything else should be identical.

    # Check that only the rewards only differ in `n` step.
    diff = np.array(rewards) != np.array(shaped_rewards)
    assert diff.sum() == n

    # TODO: Check that the differing time steps are the first `n` non-zero reward.


@pytest.mark.parametrize('n', [0, 1, 10])
def test_n_steps_delayed_start(n):
    config = {
        **ENV_CONFIG,
        'shaped_reward_config': {
            'type': 'subtask',
            'delayed_start': ('fixed', n, 'steps'),
        },
    }
    env = Env(**config)

    rewards = []
    shaped_rewards = []

    env.reset()
    for i in itertools.count():
        obs, reward, terminated, truncated, _ = env.step(ACTION_CYCLE[i % len(ACTION_CYCLE)])

        rewards.append(reward)
        shaped_rewards.append(obs['shaped_reward'].item())

        if terminated or truncated:
            break

    # Skip the first `n` steps. Everything else should be identical.

    # Check that the rewards after `n` steps are identical.
    diff = np.array(rewards) != np.array(shaped_rewards)
    assert diff[n:].sum() == 0
