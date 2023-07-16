import itertools

import torch
from minigrid.core.constants import COLOR_NAMES

from big_rl.model.model import ModularPolicy2, ModularPolicy4, ModularPolicy5, ModularPolicy5LSTM, ModularPolicy7
from big_rl.model.modular_policy_8 import ModularPolicy8
from big_rl.model.baseline import BaselineModel
from big_rl.utils import ExperimentConfigs


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
        'obs': {
            'type': 'ImageInput56',
            'config': {
                'in_channels': 1, #observation_space['obs'].shape[0]
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
    if 'obs (pseudo_reward)' in list(observation_space.keys()):
        inputs['obs (pseudo_reward)'] = {
            'type': 'ScalarInput',
            'input_mapping': ['obs (shaped_reward)', 'obs (pseudo_reward)'],
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
    elif model_type == 'ModularPolicy7':
        assert architecture is not None
        return ModularPolicy7(
                **common_model_params,
                architecture=architecture,
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
    elif model_type == 'Baseline':
        # Similar to ModularPolicy5LSTM setup
        assert architecture is not None
        for k in ['reward', 'obs (shaped_reward)']:
            if k not in inputs:
                continue
            inputs[k]['config'] = {
                    **inputs[k].get('config', {}),
                    'value_size': 1,
            }
        return BaselineModel(
                inputs=inputs,
                outputs=outputs,
                value_size=common_model_params['value_size'],
                architecture=architecture,
        ).to(device)
    raise NotImplementedError()


def env_config_presets():
    config = ExperimentConfigs()

    def init_defaults():
        ENV_NAMES = [
            'ALE/Boxing-v5',
            'ALE/VideoPinball-v5',
            'ALE/Breakout-v5',
            'ALE/StarGunner-v5',
            'ALE/Robotank-v5',
            'ALE/Atlantis-v5',
        ]
        for env_name in ENV_NAMES:
            config.add(env_name, {
                'env_name': env_name,
                'config': {
                    'frameskip': 1,
                    'full_action_space': True,
                },
                'atari_preprocessing': {
                    'screen_size': 84,
                    #'screen_size': 56,
                },
                'meta_config': {
                    'episode_stack': 1,
                    'dict_obs': True,
                    'randomize': False,
                },
            })

    init_defaults()

    return config

