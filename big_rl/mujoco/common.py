import torch

from big_rl.model.model import ModularPolicy2, ModularPolicy4, ModularPolicy5, ModularPolicy5LSTM
from big_rl.model.modular_policy_8 import ModularPolicy8
from big_rl.minigrid.common import ExperimentConfigs


def init_model(observation_space, action_space,
        model_type,
        recurrence_type,
        num_recurrence_blocks=3,
        architecture=[3,3],
        ff_size=[1024],
        hidden_size=None, # For LSTM model only
        device=torch.device('cpu')):
    observation_space = observation_space # Unused variable
    inputs = {
        'obs (state)': {
            'type': 'LinearInput',
            'config': {
                'input_size': observation_space['obs (state)'].shape[0]
            },
        },
        'reward': {
            'type': 'ScalarInput',
        },
        'action': {
            'type': 'LinearInput',
            'config': {
                'input_size': action_space.shape[0]
            },
        },
    }
    if 'obs (image)' in list(observation_space.keys()):
        inputs['obs (image)'] = {
            'type': 'ImageInput56',
            'config': {
                'in_channels': observation_space['obs (image)'].shape[0]
            },
        }
    if 'obs (reward_permutation)' in list(observation_space.keys()):
        raise NotImplementedError()
    if 'obs (shaped_reward)' in list(observation_space.keys()):
        raise NotImplementedError()
    if 'action_map' in list(observation_space.keys()):
        raise NotImplementedError()
    outputs = {
        'value': {
            'type': 'LinearOutput',
            'config': {
                'output_size': 1,
            }
        },
        'action_mean': {
            'type': 'LinearOutput',
            'config': {
                'output_size': action_space.shape[0],
            }
        },
        'action_logstd': {
            'type': 'StateIndependentOutput',
            'config': {
                'output_size': action_space.shape[0],
            }
        }
    }
    common_model_params = {
        'inputs': inputs,
        'outputs': outputs,
        'input_size': 512,
        'key_size': 512,
        'value_size': 512,
        'num_heads': 8,
        'ff_size': ff_size[0] if len(ff_size) == 1 else ff_size,
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
    elif model_type == 'ModularPolicy5LSTM':
        assert architecture is not None
        for k in ['reward', 'obs (shaped_reward)']:
            if k not in inputs:
                continue
            inputs[k]['config'] = {
                    **inputs[k].get('config', {}),
                    'value_size': 1,
            }
        return ModularPolicy5LSTM(
                inputs=inputs,
                outputs=outputs,
                value_size=common_model_params['value_size'],
                hidden_size=hidden_size,
        ).to(device)
    elif model_type == 'ModularPolicy8':
        recurrence_kwargs = {
            'ff_size': common_model_params.pop('ff_size'),
        }
        if architecture is not None:
            recurrence_kwargs['architecture'] = architecture
        return ModularPolicy8(
                **common_model_params,
                recurrence_kwargs=recurrence_kwargs,
        ).to(device)
    raise NotImplementedError()


def env_config_presets():
    config = ExperimentConfigs()

    def init_locomotion():
        config.add('locomotion-001', {
            'env_name': 'AntBaseline-v0',
            'config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
        })

        config.add_change('locomotion-002', {
            'normalize_obs': False,
        })

    def init_ant_fetch():
        config.add('fetch-001', {
            'env_name': 'AntFetch-v0',
            'normalize_obs': False,
            'config': {
                'num_objs': 1,
                'num_target_objs': 1,
            },
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
        })

        config.add_change('fetch-002', {
            'env_name': 'AntFetch-v0',
            'config': {
                'healthy_if_flipped': False,
            },
        })

        config.add_change('fetch-003', {
            'env_name': 'AntFetch-v0',
            'config': {
                'ctrl_cost_weight': 0.5/300,
                'contact_cost_weight': 5e-4/300,
                'healthy_reward': 1.0/300,
            },
        })

        # Allow the agent more time whenever it reaches a target
        config.add_change('fetch-004', {
            'env_name': 'AntFetch-v0',
            'config': {
                'max_steps_initial': 1000,
                'extra_steps_per_pickup': 500,
            },
        })

        # 0 objects
        config.add_change('fetch-004-empty', {
            'env_name': 'AntFetch-v0',
            'config': {
                'object_colours': {},
                'num_objs': 0,
                'num_target_objs': 0,
            },
        })

        # Larger environment
        for n in [5,6,7,8,9,10]:
            config.add(f'fetch-004-roomsize{n}', {
                'env_name': 'AntFetch-v0',
                'config': {
                    'room_size': n,
                },
            }, inherit='fetch-004')

        # Decrease time limit
        # It current takes around 100 steps to do a 360 and ~50 steps to run towards the object.
        config.add('fetch-005', {
            'env_name': 'AntFetch-v0',
            'config': {
                'max_steps_initial': 500,
                'extra_steps_per_pickup': 250,
            },
        }, inherit='fetch-004')

        for n in [5,6,7,8,9,10]:
            config.add(f'fetch-005-roomsize{n}', {
                'env_name': 'AntFetch-v0',
                'config': {
                    'room_size': n,
                },
            }, inherit='fetch-005')

        # Even shorter time limit
        config.add('fetch-006', {
            'env_name': 'AntFetch-v0',
            'config': {
                'max_steps_initial': 250,
                'extra_steps_per_pickup': 100,
            },
        }, inherit='fetch-004')

        for n in [5,6,7,8,9,10]:
            config.add(f'fetch-006-roomsize{n}', {
                'env_name': 'AntFetch-v0',
                'config': {
                    'room_size': n,
                },
            }, inherit='fetch-005')

    def init_arm_fetch():
        config.add('arm_fetch-001', {
            'env_name': 'ArmFetch-v0',
            'normalize_obs': False,
            'config': {
                'ctrl_cost_weight': 0.5/300,
                'num_objs': 1,
                'num_target_objs': 1,
            },
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
        })

        config.add_change('arm_fetch-002', {
            'config': {
                'num_objs': 2,
                'num_target_objs': 1,
            },
        })

        config.add_change('arm_fetch-003', {
            'config': {
                'num_objs': 2,
                'num_target_objs': 1,
                'camera_fov': 90,
            },
        })

        # Remove control cost
        config.add_change('arm_fetch-003-ctrlcost0', {
            'config': {
                'ctrl_cost_weight': 0,
            },
        })

        # Allow extra steps per pickup
        config.add('arm_fetch-004', {
            'config': {
                'max_steps_initial': 1000,
                'extra_steps_per_pickup': 500,
                'max_trials': 100,
            }
        }, inherit='arm_fetch-003')

        config.add_change('arm_fetch-004-ctrlcost0', {
            'config': {
                'ctrl_cost_weight': 0,
            },
        })

        # Easier task
        # Placed the camera further back so it's more likely to see what's happening with random movements.
        config.add('arm_fetch-005', {
            'config': {
                'max_steps_initial': 1000,
                'extra_steps_per_pickup': 500,
                'max_trials': 100,
                'camera_fov': 90,
                'camera_distance': 0,
            }
        }, inherit='arm_fetch-004')

        # Even easier task
        # One object only
        config.add('arm_fetch-006', {
            'config': {
                'num_objs': 1,
                'num_target_objs': 1,
            }
        }, inherit='arm_fetch-005')

    init_locomotion()
    init_ant_fetch()
    init_arm_fetch()

    return config
