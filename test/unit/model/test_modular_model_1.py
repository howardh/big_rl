import pytest

import torch
import gymnasium

from big_rl.model.factory import create_model


def test_save_and_load_state_dict():
    """ ... """
    batch_size = 5
    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_scalar': {
                'type': 'ScalarInput',
            },
            'foo_discrete': {
                'type': 'DiscreteInput',
                'kwargs': {
                    'input_size': 3,
                },
            }
        },
        'output_modules': {
            'some_output': {
                'type': 'LinearOutput',
                'kwargs': {
                    'output_size': 3,
                },
            },
            'other_output': {
                'type': 'StateIndependentOutput',
                'kwargs': {
                    'output_size': 3,
                },
            },
        },
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }
    model1 = create_model(config)
    model2 = create_model(config)
    model2.load_state_dict(model1.state_dict())

    # Verify that the model can run without errors
    hidden = model1.init_hidden(batch_size)
    for _ in range(10):  # With inputs omitted
        inputs = {
            'foo_scalar': torch.randn(batch_size, 1),
            'foo_discrete': torch.randint(0, 3, (batch_size, )),
        }
        output1 = model1(inputs, hidden)
        output2 = model2(inputs, hidden)

        assert torch.equal(output1['some_output'], output2['some_output'])
        assert torch.equal(output1['other_output'], output2['other_output'])
        for h1,h2 in zip(output1['hidden'], output2['hidden']):
            assert torch.equal(h1, h2)

        hidden = output1['hidden']


def test_save_and_load_state_dict_2():
    """ With core container modules """
    batch_size = 5
    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_scalar': {
                'type': 'ScalarInput',
            },
            'foo_discrete': {
                'type': 'DiscreteInput',
                'kwargs': {
                    'input_size': 3,
                },
            }
        },
        'output_modules': {
            'some_output': {
                'type': 'LinearOutput',
                'kwargs': {
                    'output_size': 3,
                },
            },
            'other_output': {
                'type': 'StateIndependentOutput',
                'kwargs': {
                    'output_size': 3,
                },
            },
        },
        'core_modules': {
            'container': 'parallel',
            'modules': [{
                'type': 'RecurrentAttention17',
            }, {
                'type': 'RecurrentAttention17',
            }],
        },
    }
    model1 = create_model(config)
    model2 = create_model(config)
    model2.load_state_dict(model1.state_dict())

    # Verify that the model can run without errors
    hidden = model1.init_hidden(batch_size)
    for _ in range(10):  # With inputs omitted
        inputs = {
            'foo_scalar': torch.randn(batch_size, 1),
            'foo_discrete': torch.randint(0, 3, (batch_size, )),
        }
        output1 = model1(inputs, hidden)
        output2 = model2(inputs, hidden)

        assert torch.equal(output1['some_output'], output2['some_output'])
        assert torch.equal(output1['other_output'], output2['other_output'])
        for h1,h2 in zip(output1['hidden'], output2['hidden']):
            assert torch.equal(h1, h2)

        hidden = output1['hidden']
