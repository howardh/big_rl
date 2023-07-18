import pytest

import torch

from big_rl.model.model import ModularPolicy5, ModularPolicy5LSTM


def _init_model(model_cls, inputs, outputs, **kwargs):
    if model_cls is ModularPolicy5:
        params = {
            'input_size': 8,
            'key_size': 8,
            'value_size': 8,
            'num_heads': 1,
            'ff_size': 8,
            'architecture': (2, 3),
            'recurrence_type': 'RecurrentAttention14',
            **kwargs,
        }
    elif model_cls is ModularPolicy5LSTM:
        params = {
            'value_size': 8,
            'hidden_size': 8,
            **kwargs,
        }
    else:
        raise ValueError(f'Unknown model class {model_cls}')
    return model_cls(
        inputs=inputs,
        outputs=outputs,
        **params,
    )


@pytest.mark.parametrize('recurrence_type', ['RecurrentAttention11', 'RecurrentAttention14'])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_empty_input(recurrence_type, batch_size):
    model = _init_model(ModularPolicy5, {}, {}, recurrence_type=recurrence_type)

    hidden = model.init_hidden(batch_size)

    model({}, hidden)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_lstm_no_output(batch_size):
    model = _init_model(
        ModularPolicy5LSTM,
        inputs={
            'obs (image)': {
                'type': 'ImageInput56',
                'config': {
                        'in_channels': 3,
                },
            },
            'reward': {
                'type': 'ScalarInput',
                'config': {
                        'value_size': 1,
                },
            },
            'action': {
                'type': 'DiscreteInput',
                'config': {
                        'input_size': 5,
                },
            },
        },
        outputs={}
    )

    hidden = model.init_hidden(batch_size)

    obs = {
        'obs (image)': torch.randn(batch_size, 3, 56, 56),
        'reward': torch.randn(batch_size, 1),
        'action': torch.randint(0, 5, (batch_size,)),
    }
    model(obs, hidden)


@pytest.mark.parametrize('batch_size', [2, 3])
def test_lstm(batch_size):
    model = _init_model(
        ModularPolicy5LSTM,
        inputs={
            'obs (image)': {
                'type': 'ImageInput56',
                'config': {
                        'in_channels': 3,
                },
            },
            'reward': {
                'type': 'ScalarInput',
                'config': {
                        'value_size': 1,
                },
            },
            'action': {
                'type': 'DiscreteInput',
                'config': {
                        'input_size': 5,
                },
            },
        },
        outputs={
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
        hidden_size=7,
    )

    hidden = model.init_hidden(batch_size)

    obs = {
        'obs (image)': torch.randn(batch_size, 3, 56, 56),
        'reward': torch.randn(batch_size, 1),
        'action': torch.randint(0, 5, (batch_size,)),
    }
    model(obs, hidden)
