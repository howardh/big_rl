import pytest

import gymnasium

import big_rl.mujoco.envs


ENV_NAMES = [
    'AntForward-v4',
    'AntBackward-v4',
    'AntNoVelocity-v4',
    #'AntNoVelocityForward-v4',
    #'AntNoVelocityBackward-v4',
    'Ball-v4',
    'BallNoVelocityForward-v4',
    'BallNoVelocityBackward-v4',
]


@pytest.mark.parametrize('env_name', ENV_NAMES)
def test_no_error(env_name):
    # Test that we can reset and run a few steps without error

    env = gymnasium.make(env_name)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())


@pytest.mark.parametrize('env_name', ENV_NAMES)
def test_obs_space_matches_obs(env_name):
    # Check that the observation space matches the observation receives from `reset` and `step`

    env = gymnasium.make(env_name)
    obs, _ = env.reset()
    for _ in range(10):
        if isinstance(env.observation_space, gymnasium.spaces.Box):
            assert obs.shape == env.observation_space.shape
        else:
            raise NotImplementedError()

        obs, _, _, _, _ = env.step(env.action_space.sample())
