import copy
from typing import Sequence, Iterable, Union, Tuple, Mapping

import gymnasium
import gymnasium.spaces
import torch

from big_rl.model.model import ModularPolicy2, ModularPolicy4, ModularPolicy5


def init_model(observation_space, action_space,
        model_type,
        recurrence_type,
        num_recurrence_blocks=3,
        architecture=[3,3],
        device=torch.device('cpu')):
    observation_space = observation_space # Unused variable
    inputs = {
        'obs (image)': {
            'type': 'ImageInput56',
            'config': {
                'in_channels': observation_space['obs (image)'].shape[0]
            },
        },
        'reward': {
            'type': 'ScalarInput',
        },
        'action': {
            'type': 'DiscreteInput',
            'config': {
                'input_size': action_space.n
            },
        },
    }
    # XXX: The membership test behaviour changes between gym and gymnasium. If any libraries are updated, make sure that this still works.
    if 'obs (reward_permutation)' in list(observation_space.keys()):
        inputs['obs (reward_permutation)'] = {
            'type': 'LinearInput',
            'config': {
                'input_size': observation_space['obs (reward_permutation)'].shape[0]
            }
        }
    if 'obs (shaped_reward)' in list(observation_space.keys()):
        inputs['obs (shaped_reward)'] = {
            'type': 'ScalarInput',
        }
    if 'action_map' in list(observation_space.keys()):
        inputs['action_map'] = {
            'type': 'MatrixInput',
            'config': {
                'input_size': list(observation_space['action_map'].shape),
                'num_heads': 8,
            }
        }
    outputs = {
        'value': {
            'type': 'LinearOutput',
            'config': {
                'output_size': 1,
            }
        },
        'action': {
            'type': 'LinearOutput',
            'config': {
                'output_size': action_space.n,
            }
        },
    }
    common_model_params = {
        'inputs': inputs,
        'outputs': outputs,
        'input_size': 512,
        'key_size': 512,
        'value_size': 512,
        'num_heads': 8,
        'ff_size': 1024,
        'recurrence_type': recurrence_type,
    }
    if model_type == 'ModularPolicy2':
        return ModularPolicy2(
                **common_model_params,
                num_blocks=num_recurrence_blocks,
        ).to(device)
    elif model_type == 'ModularPolicy4':
        assert architecture is not None
        return ModularPolicy4(
                **common_model_params,
                architecture=architecture,
        ).to(device)
    elif model_type == 'ModularPolicy5':
        assert architecture is not None
        return ModularPolicy5(
                **common_model_params,
                architecture=architecture,
        ).to(device)
    raise NotImplementedError()


def merge_space(*spaces):
    new_space = {}
    for space in spaces:
        for k,v in space.items():
            if k in new_space:
                assert new_space[k] == v, f"Space mismatch for key {k}: {new_space[k]} != {v}"
            else:
                new_space[k] = v
    return gymnasium.spaces.Dict(new_space)


def zip2(*args) -> Iterable[Union[Tuple,Mapping]]:
    """
    Zip objects together. If dictionaries are provided, the lists within the dictionary are zipped together.

    >>> list(zip2([1,2,3], [4,5,6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(zip2({'a': [4,5,6], 'b': [7,8,9]}))
    [{'a': 4, 'b': 7}, {'a': 5, 'b': 8}, {'a': 6, 'b': 9}]

    >>> list(zip2([1,2,3], {'a': [4,5,6], 'b': [7,8,9]}))
    [(1, {'a': 4, 'b': 7}), (2, {'a': 5, 'b': 8}), (3, {'a': 6, 'b': 9})]

    >>> import torch
    >>> list(zip2(torch.tensor([1,2,3]), torch.tensor([4,5,6])))
    [(tensor(1), tensor(4)), (tensor(2), tensor(5)), (tensor(3), tensor(6))]
    """
    if len(args) == 1:
        if isinstance(args[0],(Sequence)):
            return args[0]
        if isinstance(args[0],torch.Tensor):
            return (x for x in args[0])
        if isinstance(args[0], dict):
            keys = args[0].keys()
            return (dict(zip(keys, vals)) for vals in zip(*(args[0][k] for k in keys)))
    return zip(*[zip2(a) for a in args])


def merge(source, destination):
    """
    (Source: https://stackoverflow.com/a/20666342/382388)

    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    destination = copy.deepcopy(destination)
    if isinstance(source, Mapping):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                destination[key] = merge(value, node)
            elif isinstance(value, list):
                if isinstance(destination[key],list):
                    destination[key] = [merge(s,d) for s,d in zip(source[key],destination[key])]
                else:
                    destination[key] = value
            else:
                destination[key] = value

        return destination
    else:
        return source


class ExperimentConfigs(dict):
    def __init__(self):
        self._last_key = None
    def add(self, key, config, inherit=None):
        if key in self:
            raise Exception(f'Key {key} already exists.')
        if inherit is None:
            self[key] = config
        else:
            self[key] = merge(config,self[inherit])
        self._last_key = key
    def add_change(self, key, config):
        self.add(key, config, inherit=self._last_key)


class ConfigReplace:
    def __init__(self, value):
        self.value = value


class ConfigMerge:
    def __init__(self, value):
        self.value = value


def env_config_presets():
    config = ExperimentConfigs()

    def init_fetch():
        config.add('fetch-debug', {
            'env_name': 'MiniGrid-MultiRoom-v1',
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 1,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 5,
                'max_room_size': 5,
                'door_prob': 0.5,
                'max_steps_multiplier': 5,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_types': 1,
                    'num_obj_colors': 2,
                    'prob': 1.0, # 0.0 chance of flipping the reward
                },
                #'task_randomization_prob': 0.02, # 86% chance of happening at least once, with a 50% change of the randomized task being unchanged.
            }
        })
        config.add('fetch-001', {
            'env_name': 'MiniGrid-MultiRoom-v1',
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 5,
                'max_room_size': 5,
                'door_prob': 0.5,
                'max_steps_multiplier': 5,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_types': 1,
                    'num_obj_colors': 2,
                    'prob': 1.0, # 0.0 chance of flipping the reward
                },
            }
        })
        config.add_change('fetch-002', {
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 8,
                'max_room_size': 16,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_types': 2,
                    'num_obj_colors': 6,
                },
            }
        })
        config.add_change('fetch-002-shaped', {
            'config': {
                'shaped_reward_config': {
                    'type': 'inverse distance',
                },
            }
        })
        config.add_change('fetch-002-shaped-adjacent', {
            'config': {
                'shaped_reward_config': {
                    'type': 'adjacent to subtask',
                },
            }
        })
        # Skipping 003 to match up with the delayed task numbering
        config.add('fetch-002-pbrs', {
            'config': {
                'reward_type': 'pbrs',
                'pbrs_scale': 0.1,
            }
        }, inherit='fetch-002')

    def init_delayed():
        config.add('delayed-001', {
            'env_name': 'MiniGrid-Delayed-Reward-v0',
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 5,
                'max_room_size': 6,
                'door_prob': 0.5,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_colors': 6,
                    'prob': 1.0, # 0.0 chance of flipping the reward
                },
            }
        })
        config.add_change('delayed-002', {
            # Looks like I forgot to make changes?
        })
        config.add_change('delayed-003', {
            'config': {
                'min_room_size': 8,
                'max_room_size': 16,
            }
        })
        config.add_change('delayed-003-shaped', {
            'config': {
                'shaped_reward_config': {
                    'type': 'inverse distance',
                },
            }
        })
        config.add_change('delayed-003-shaped-adjacent', {
            'config': {
                'shaped_reward_config': {
                    'type': 'adjacent to subtask',
                },
            }
        })

    init_fetch()
    init_delayed()

    return config

