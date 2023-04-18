import pytest

import torch

from big_rl.model.modular_policy_8 import ModularPolicy8


def _init_model(inputs, outputs, **kwargs):
    params = {
            'input_size': 8,
            'key_size': 8,
            'value_size': 8,
            'num_heads': 1,
            'recurrence_type': 'RecurrentAttention16',
            'recurrence_kwargs': {
                'ff_size': 8,
                'architecture': (2,3),
            },
            **kwargs,
    }
    return ModularPolicy8(
        inputs = inputs,
        outputs = outputs,
        **params,
    )


@pytest.mark.parametrize('recurrence_type', ['RecurrentAttention16'])
@pytest.mark.parametrize('batch_size', [1,2])
def test_empty_input(recurrence_type, batch_size):
    model = _init_model({}, {}, recurrence_type=recurrence_type)

    hidden = model.init_hidden(batch_size)

    model({}, hidden)


@pytest.mark.parametrize('batch_size', [2,3])
def test_call(batch_size):
    model = _init_model(
            inputs = {
                'obs (image)': {
                    'type': 'ImageInput56',
                    'config': {
                        'in_channels': 3,
                    },
                },
                'reward': {
                    'type': 'ScalarInput',
                },
                'action': {
                    'type': 'DiscreteInput',
                    'config': {
                        'input_size': 5,
                    },
                },
            },
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
                        'output_size': 5,
                    }
                },
            },
    )

    hidden = model.init_hidden(batch_size)

    obs = {
        'obs (image)': torch.randn(batch_size, 3, 56, 56),
        'reward': torch.randn(batch_size, 1),
        'action': torch.randint(0, 5, (batch_size,)),
    }
    model(obs, hidden)


@pytest.mark.parametrize('batch_size', [3])
def test_input_mapping(batch_size):
    """ Test the ability to remap inputs to different modules. """
    model = _init_model(
            inputs = {
                'reward': {
                    'type': 'ScalarInput',
                    'input_mapping': ['reward', 'reward2'], # Specify that both 'reward' and 'reward2' should be mapped to this input module.
                },
            },
            outputs = {
                'action': {
                    'type': 'LinearOutput',
                    'config': {
                        'output_size': 50,
                    }
                },
            },
    )

    hidden = model.init_hidden(batch_size)

    obs = {
        'reward': torch.randn(batch_size, 1),
    }
    output1 = model(obs, hidden)

    obs = {
        'reward2': obs['reward'],
    }
    output2 = model(obs, hidden)

    assert output1.keys() == output2.keys()
    assert torch.allclose(output1['action'], output2['action'])
