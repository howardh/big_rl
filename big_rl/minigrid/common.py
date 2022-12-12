import copy
import os
import time
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

##################################################
# File IO
##################################################


def get_results_root_directory(temp=False):
    """ Get the directory where results are to be stored. """
    host_name = os.uname()[1]
    # Mila (Check available resources with `sinfo`)
    mila_hostnames = ['rtx', 'leto', 'eos', 'bart', 'mila', 'kepler', 'power', 'apollor', 'apollov', 'cn-', 'login-1', 'login-2', 'login-3', 'login-4']
    if host_name.endswith('server.mila.quebec') or any((host_name.startswith(n) for n in mila_hostnames)):
        if temp:
            return "/miniscratch/huanghow"
        else:
            return "/network/projects/h/huanghow"
    # RL Lab
    if host_name == "agent-server-1" or host_name == "agent-server-2":
        return "/home/ml/users/hhuang63/results"
        #return "/NOBACKUP/hhuang63/results"
    if host_name == "garden-path" or host_name == "ppl-3":
        return "/home/ml/hhuang63/results"
    # Compute Canada
    if host_name.find('gra') == 0 or host_name.find('cdr') == 0:
        return "/home/hhuang63/scratch/results"
    # Local
    if host_name.find('howard-pc') == 0:
        return "/home/howard/tmp/results"
    # Travis
    if host_name.startswith('travis-'):
        return './tmp'
    raise NotImplementedError("No default path defined for %s" % host_name)


def is_slurm():
    return 'SLURM_JOB_ID' in os.environ


def generate_id(slurm_split: bool = False) -> str:
    """ Generate an identifier that is unique to the current run. If the script is run as a Slurm job, the job ID is used. Otherwise, the current date and time are used.

    Args:
        slurm_split (bool): Set to True if the Slurm job is a single job split into multiple parts as an array job. This will give the same identifier to all jobs of the array.
    """
    slurm_job_id = os.environ.get('SLURM_JOB_ID') # Unique per job
    slurm_array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID') # Same for every job in an array
    slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID') # Unique per job in the array

    if slurm_job_id is not None: # If it's a slurm job, use the job ID in the directory name
        if slurm_array_job_id is not None:
            # `slurm_array_job_id` is None if it is not an array job
            if slurm_split:
                # If `slurm_split` is True, that means we want to run one experiment split over multiple jobs in an array, so every job in the array should have the same `trial_id`.
                return f'{slurm_array_job_id}'
            else:
                return f'{slurm_array_job_id}_{slurm_array_task_id}'
        else:
            return f'{slurm_job_id}'
    else: # If it is not a slurm job, use the data/time to name the directory
        return time.strftime("%Y_%m_%d-%H_%M_%S")


def create_unique_file(directory, name, extension):
    """ Create a unique file in the given directory with the given name and extension. If the file already exists, a number is appended to the name.

    Returns the full path to the file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    if extension.startswith('.'):
        extension = extension[1:]
    filename = os.path.join(directory, f'{name}.{extension}')
    index = 0
    while True:
        try: 
            # Create the file if it doesn't exist
            # This is atomic on POSIX systems (https://linux.die.net/man/3/open says "The check for the existence of the file and the creation of the file if it does not exist shall be atomic with respect to other threads executing open() naming the same filename in the same directory with O_EXCL and O_CREAT set")
            # XXX: Not sure if this is atomic on a non-POSIX filesystem
            f = os.open(filename,  os.O_CREAT | os.O_EXCL)
            os.close(f)
        except FileExistsError:
            index += 1
            filename = os.path.join(directory, f'{name}-{index}.{extension}')
            continue
        return filename


def create_unique_directory(directory, name):
    """ Create a unique subdirectory in the given directory with the given name. If the directory already exists, a number is appended to the name.

    Returns the full path to the new subdirectory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f'{name}')
    index = 0
    while True:
        try: 
            # Create the directory if it doesn't exist
            # XXX: Not sure about atomicity of this operation
            os.mkdir(filename)
        except FileExistsError:
            index += 1
            filename = os.path.join(directory, f'{name}-{index}')
            continue
        return filename


##################################################
# Experiment Configs
##################################################


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
    if not isinstance(source, type(destination)):
        return source
    if isinstance(source, Mapping):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                destination[key] = merge(value, node)
            elif isinstance(value, list):
                if isinstance(destination[key],list) and len(destination[key]) == len(value):
                    destination[key] = [merge(s,d) for s,d in zip(source[key],destination[key])]
                else:
                    destination[key] = value
            elif isinstance(value, ConfigReplace):
                destination[key] = value.value
            elif isinstance(value, ConfigDelete):
                del destination[key]
            else:
                destination[key] = value

        return destination
    else:
        return source


class ExperimentConfigs(dict):
    """ A dictionary of experiment configurations. The main purpose of this class is to allow configs to be written in a way where it is easier to see the differences from one set of configs to another, and to avoid bugs arising from copy-pasting configurations and forgetting to change their names.

    New configurations can be added as complete dictionaries, or as partial dictionaries that are merged with the existing configurations. By default, dicts and lists of dicts (if both are the same length) are merged recursively. All other types are completely replaced with the new value. This behaviour can be overridden by wrapping the object with `ConfigMerge` or `ConfigReplace`. Keys can also be deleted with `ConfigDelete`.
    """
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
        assert isinstance(value,dict)
        self.value = value


class ConfigDelete:
    ...


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

        # Noisy shaped rewards
        config.add('fetch-002-shaped-noisy-debug', {
            'config': {
                'shaped_reward_config': {
                    'type': 'adjacent to subtask',
                    'noise': ('stop', 500, 'steps'),
                },
            }
        }, inherit='fetch-002')

        # Skipping 003 to match up with the delayed task numbering
        config.add('fetch-004', {
            'config': {
                'min_room_size': 4,
                'max_room_size': 6,
                'reward_type': 'standard',
            }
        }, inherit='fetch-002')
        config.add_change('fetch-004-pbrs', {
            'config': {
                'reward_type': 'pbrs',
                'pbrs_scale': 0.1,
                'pbrs_discount': 0.99,
            }
        })
        config.add('fetch-004-shaped', {
            'config': {
                'shaped_reward_config': {
                    'type': 'inverse distance',
                },
            }
        }, inherit='fetch-004')

        # Increase map size from 4-6 to 5-8
        config.add('fetch-004-bigger', {
            'config': {
                'min_room_size': 5,
                'max_room_size': 12,
            }
        }, inherit='fetch-004')
        config.add('fetch-004-bigger-pbrs', {
            'config': {
                'min_room_size': 5,
                'max_room_size': 12,
            }
        }, inherit='fetch-004-pbrs')
        config.add('fetch-004-bigger-shaped', {
            'config': {
                'min_room_size': 5,
                'max_room_size': 12,
            }
        }, inherit='fetch-004-shaped')

        # Remove reward signal, keep shaped reward, but cut off shaped reward after some number of steps
        for cutoff in [500, 200, 100, 50, 20, 1]:
            config.add(f'fetch-005-stop_{cutoff}', {
                'meta_config': {
                    'include_reward': False,
                },
                'config': {
                    'min_room_size': 5,
                    'max_room_size': 12,
                    'shaped_reward_config': {
                        'type': 'subtask',
                        'noise': ('stop', cutoff, 'steps'),
                    },
                }
            }, inherit='fetch-004')
            for delay in [1]:
                config.add(f'fetch-005-stop_{cutoff}-delay_{delay}', {
                    'meta_config': {
                        'include_reward': False,
                    },
                    'config': {
                        'min_room_size': 5,
                        'max_room_size': 12,
                        'shaped_reward_config': {
                            'type': 'subtask',
                            'noise': ('stop', cutoff, 'steps'),
                            'delay': ('fixed', delay)
                        },
                    }
                }, inherit='fetch-004')
            for delay in [(1,2)]:
                config.add(f'fetch-005-stop_{cutoff}-delay_{delay[0]}_{delay[1]}', {
                    'meta_config': {
                        'include_reward': False,
                    },
                    'config': {
                        'min_room_size': 5,
                        'max_room_size': 12,
                        'shaped_reward_config': {
                            'type': 'subtask',
                            'noise': ('stop', cutoff, 'steps'),
                            'delay': ('random', delay[0], delay[1])
                        },
                    }
                }, inherit='fetch-004')
        for cutoff in [100, 50, 0]:
            config.add(f'fetch-005-stop_{cutoff}_trials', {
                'meta_config': {
                    'include_reward': False,
                },
                'config': {
                    'min_room_size': 5,
                    'max_room_size': 12,
                    'shaped_reward_config': {
                        'type': 'subtask',
                        'noise': ('stop', cutoff, 'trials'),
                    },
                }
            }, inherit='fetch-004')
        config.add(f'fetch-005-zero_1_1_trials', {
            'meta_config': {
                'include_reward': False,
            },
            'config': {
                'min_room_size': 5,
                'max_room_size': 12,
                'shaped_reward_config': {
                    'type': 'subtask',
                    'noise': ('zero', (1,1), 'cycle_trials'),
                },
            }
        }, inherit='fetch-004')


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

        # Noisy shaped rewards
        for cutoff in [500, 200, 100, 1, 0]:
            config.add(f'delayed-003-shaped_adjacent-stop_noise_{cutoff}', {
                'config': {
                    'shaped_reward_config': {
                        'type': 'adjacent to subtask',
                        'noise': ('stop', cutoff, 'steps'),
                    },
                }
            }, inherit='delayed-003')

    init_fetch()
    init_delayed()

    return config

