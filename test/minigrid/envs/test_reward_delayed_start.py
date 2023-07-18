import pytest

import numpy as np

from big_rl.minigrid.envs import RewardDelayedStart


@pytest.mark.parametrize('units', ['steps', 'trials'])
def test_no_delay(units):
    transform = RewardDelayedStart(delay_type='fixed', start_when=0, units=units)

    for _ in range(10):
        reward = np.random.uniform()
        assert transform(reward) == reward


def test_1_step_delay():
    transform = RewardDelayedStart(delay_type='fixed', start_when=1, units='steps')

    reward = np.random.uniform()
    assert transform(reward) == 0
    for _ in range(10):
        reward = np.random.uniform()
        assert transform(reward) == reward


def test_1_trial_delay():
    transform = RewardDelayedStart(delay_type='fixed', start_when=1, units='trials')

    for _ in range(10):
        reward = np.random.uniform()
        assert transform(reward) == 0
    transform.trial_finished()
    for _ in range(10):
        reward = np.random.uniform()
        assert transform(reward) == reward


def test_1_to_3_step_delay():
    transform = RewardDelayedStart(delay_type='random', start_when=(1, 3), units='steps')

    reward = np.random.uniform()
    assert transform(reward) == 0

    for _ in range(3):
        transform(0)

    for _ in range(10):
        reward = np.random.uniform()
        assert transform(reward) == reward
