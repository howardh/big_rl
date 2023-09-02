import pytest
import itertools

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.mock_agent import mock_agent

TASK_CONFIG = {
    'args': {
         'pseudo_reward_config': {
             'type': 'subtask'
         },
         'reward_correct': 1,
         'reward_flip_prob': 0,
         'reward_incorrect': -1,
         'reward_type': 'standard'
    },
    'task': 'fetch',
    'wrappers': [],
}
ENV_CONFIG = {
    'min_num_rooms': 1,
    'max_num_rooms': 1,
    'min_room_size': 4,
    'max_room_size': 4,
    'num_obj_colors': 2,
    'num_obj_types': 1,
    'num_objs': 2,
    'num_trials': 10,
    'task_config': {
        **TASK_CONFIG,
    }
}


@pytest.mark.skip(reason="No test. Just putting this here as a reference.")
def test_stuff():
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'args': {
                     'pseudo_reward_config': {
                         'type': 'subtask'
                     },
                     'reward_correct': 1,
                     'reward_flip_prob': 0,
                     'reward_incorrect': -1,
                     'reward_type': 'standard'
                },
                'task': 'fetch',
                'wrappers': [{
                    'type': 'pseudo_reward_cutoff',
                    'args': {
                        'threshold_resume': (4, 'trial failure', 10),
                        'threshold_stop': (9, 'trial success', 10)
                    },
                },{
                    'type': 'random_reset',
                    'args': {'prob': 0.02},
                },{
                    'type': 'pseudo_reward_delay',
                    'args': {
                        'delay_type': 'random',
                        'overlap': 'replace',
                        'steps': (1, 5)
                    },
                }]
            }
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    total_reward = 0
    for _, reward, _, _, _ in mock_agent(env, itertools.cycle([('key', 'green'), ('key', 'blue')])):
        total_reward += float(reward)
    assert total_reward == 0


def test_pseudo_reward_delay_1():
    """ Delay the pseudo-reward by 1 step. Check that the pseudo-reward matches the reward with that 1 step delay. """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'args': {
                     'pseudo_reward_config': {
                         'type': 'subtask'
                     },
                     'reward_correct': 1,
                     'reward_flip_prob': 0,
                     'reward_incorrect': -1,
                     'reward_type': 'standard'
                },
                'task': 'fetch',
                'wrappers': [{
                    'type': 'pseudo_reward_delay',
                    'args': {
                        'delay_type': 'random',
                        'overlap': 'replace',
                        'steps': (1, 1)
                    },
                }]
            }
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    history = list(mock_agent(env, itertools.cycle([('key', 'green'), ('key', 'blue')])))
    for x1,x2 in zip(history, history[1:]):
        _, reward, _, _, _ = x1
        obs, _, _, _, _ = x2
        if reward != 0:
            assert obs['obs (pseudo_reward)'] == reward


def test_random_reset():
    """ Reset the environment after every trial. The target object is randomised at every trial, which means it is very unlikely that it is the same target every time. The probability of it being the same for every trial in 10 trials is 0.5^10 < 0.001. """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'args': {
                     'pseudo_reward_config': {
                         'type': 'subtask'
                     },
                     'reward_correct': 1,
                     'reward_flip_prob': 0,
                     'reward_incorrect': -1,
                     'reward_type': 'standard'
                },
                'task': 'fetch',
                'wrappers': [{
                    'type': 'random_reset',
                    'args': {
                        'prob': 1.0, # 100% chance of resetting after each trial
                    },
                }]
            }
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    total_reward = 0
    reward_count = 0
    history = list(mock_agent(env, itertools.cycle([('key', 'green')])))
    for _,reward,_,_,_  in history:
        total_reward += float(reward)
        if reward != 0:
            reward_count += 1
    assert reward_count == 10
    assert total_reward != -10 and total_reward != 10


def test_pseudo_reward_cutoff():
    """ Set the cut-off threshold to 3 out of 3 correct trials with the agent choosing the correct object every time. The pseudo-reward should total to exactly 3. """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'args': {
                     'pseudo_reward_config': {
                         'type': 'subtask'
                     },
                     'reward_correct': 1,
                     'reward_flip_prob': 0,
                     'reward_incorrect': -1,
                     'reward_type': 'standard',
                     'fixed_target': ('key', 'green'),
                },
                'task': 'fetch',
                'wrappers': [{
                    'type': 'pseudo_reward_cutoff',
                    'args': {
                        'threshold_stop': (3, 'trial success', 3),
                    },
                }]
            }
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    history = list(mock_agent(env, itertools.cycle([('key', 'green')])))
    total_reward = 0
    total_pseudo_reward = 0
    for obs,reward,_,_,_ in history:
        total_reward += float(reward)
        total_pseudo_reward += obs['obs (pseudo_reward)']

    assert total_reward == 10
    assert total_pseudo_reward == 3


def test_alternating_tasks_3():
    """ Set the task to alternate after every 3 trials, regardless of whether it they're successful or not. The mock agent is set to always choose the first target. This means it should get 3 correct, 3 incorrect, 3 correct, and 1 incorrect for a total reward of 3-3+3-1=2. """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'task': 'alternating',
                'args': {
                    'task_duration': (3, 'trials'),
                    'randomize_tasks': False,
                    'tasks': [{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'green'),
                        },
                    },{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'blue'),
                        },
                    }]
                }
            },
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    history = list(mock_agent(env, itertools.cycle([('key', 'green')])))
    total_reward = 0
    for _,reward,_,_,_ in history:
        total_reward += float(reward)
    assert total_reward == 2


def test_alternating_tasks_5():
    """ Set the task to alternate after every 5 trials, regardless of whether they're successful or not. The mock agent is set to get everything correct. Verify that it receives a final reward of 10 """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'task': 'alternating',
                'args': {
                    'task_duration': (5, 'trials'),
                    'randomize_tasks': False,
                    'tasks': [{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'green'),
                        },
                    },{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'blue'),
                        },
                    }]
                }
            },
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    history = list(mock_agent(env, [('key', 'green')]*5 + [('key', 'blue')]*5))
    total_reward = 0
    for _,reward,_,_,_ in history:
        total_reward += float(reward)
    assert total_reward == 10


def test_alternating_tasks_5_success():
    """ Set the task to alternate after every 5 successful trials. The mock agent is set to get everything wrong. Verify that it receives a final reward of -10 """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'task': 'alternating',
                'args': {
                    'task_duration': (5, 'trial success'),
                    'randomize_tasks': False,
                    'tasks': [{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'green'),
                        },
                    },{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'blue'),
                        },
                    }]
                }
            },
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    history = list(mock_agent(env, [('key', 'blue')]*10))
    total_reward = 0
    for _,reward,_,_,_ in history:
        total_reward += float(reward)
    assert total_reward == -10


def test_alternating_tasks_50():
    """ Debugging thing """
    config = {
        'env_name': 'MiniGrid-MultiRoom-v2',
        'config': {
            **ENV_CONFIG,
            'task_config': {
                'task': 'alternating',
                'args': {
                    'task_duration': (50, 'trial success'),
                    'randomize_tasks': False,
                    'tasks': [{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'green'),
                        },
                    },{
                        'task': 'fetch',
                        'args': {
                            'pseudo_reward_config': {
                                'type': 'subtask',
                            },
                            'reward_correct': 1,
                            'reward_flip_prob': 0,
                            'reward_incorrect': -1,
                            'reward_type': 'standard',
                            'fixed_target': ('key', 'blue'),
                        },
                    }]
                }
            },
        },
        'meta_config': {
            'dict_obs': True,
            'episode_stack': 1,
            'include_reward': True,
            'randomize': False
        },
        'minigrid_config': {}
    }
    env = make_env(**config)
    history = list(mock_agent(env, [('key', 'blue')]*10))
    total_reward = 0
    for _,reward,_,_,_ in history:
        total_reward += float(reward)
    assert total_reward == -10
