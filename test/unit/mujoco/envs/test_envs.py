import pytest

import gymnasium

import big_rl.mujoco.envs


ENV_NAMES = [
    'AntForward-v4',
    'AntBackward-v4',
    'AntNoVelocity-v4',
    #'AntNoVelocityForward-v4',
    #'AntNoVelocityBackward-v4',
]


@pytest.mark.parametrize('env_name', ENV_NAMES)
def test_no_error(env_name):
    # Test that we can reset and run a few steps without error

    env = gymnasium.make(env_name)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
