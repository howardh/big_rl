import pytest

import torch
from big_rl.model.core_module.container import CoreModule
from big_rl.model.input_module.base import InputModule
from big_rl.model.input_module.container import InputModuleContainer
from big_rl.model.modular_model_1 import ModularModel1
from big_rl.model.output_module.base import OutputModule
from big_rl.model.output_module.container import OutputModuleContainer
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


# Test hidden states


class DummyRecurrentModule:
    def __init__(self, hidden_size: list[tuple[int, ...]]):
        self._n_hidden = len(hidden_size)
        self._hidden_size = hidden_size
        self._hidden = [
            torch.randn(1, *h)
            for h in hidden_size
        ]

    @property
    def n_hidden(self) -> int:
        return self._n_hidden

    def init_hidden(self, batch_size: int) -> tuple:
        return tuple([
            h.expand(batch_size, *h.shape[1:])
            for h in self._hidden
        ])

    def _check_hidden(self, hidden):
        for h1, h2 in zip(hidden, self._hidden):
            assert h1.shape[1:] == h2.shape[1:]
            assert ((h1 - h2) == 0).all() # Using subtraction so that it broadcasts automatically

class DummyRecurrentInputModule(DummyRecurrentModule, InputModule):
    def __init__(self, input_size: int, key_size: int, value_size: int, hidden_size: list[tuple[int, ...]] = []):
        InputModule.__init__(self)
        self._input_size = input_size
        self._key_size = key_size
        self._value_size = value_size

        DummyRecurrentModule.__init__(self, hidden_size)

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        self._check_hidden(hidden)
        return {
            'key': torch.randn(batch_size, self._key_size),
            'value': torch.randn(batch_size, self._value_size),
            'hidden': hidden,
        }


class DummyCoreModule(DummyRecurrentModule, CoreModule):
    def __init__(self, key_size, value_size, num_heads, hidden_size: list[tuple[int, ...]]=[]):
        CoreModule.__init__(self)
        DummyRecurrentModule.__init__(self, hidden_size)

    def forward(self, key, value, hidden):
        self._check_hidden(hidden)
        return {
            'key': key,
            'value': value,
            'hidden': hidden,
        }


class DummyRecurrentOutputModule(DummyRecurrentModule, OutputModule):
    def __init__(self, key_size: int, value_size: int, num_heads: int, hidden_size: list[tuple[int, ...]] = []):
        OutputModule.__init__(self)
        DummyRecurrentModule.__init__(self, hidden_size)

    def forward(self, key, value, hidden):
        self._check_hidden(hidden)
        return {
            'output': torch.randn(key.shape[0], 3),
            'hidden': hidden,
        }


def test_recurrence():
    input_size = 5
    key_size = 3
    value_size = 2
    num_heads = 1
    common_params = {
        'key_size': key_size,
        'value_size': value_size,
    }
    model = ModularModel1(
        input_modules=InputModuleContainer(torch.nn.ModuleDict({
            'foo': DummyRecurrentInputModule(
                input_size=input_size,
                **common_params,
                hidden_size=[(1, 2), (3, 4)],
            ),
        })),
        core_modules=DummyCoreModule(
            **common_params,
            num_heads=num_heads,
            hidden_size=[(5, 6)],
        ),
        output_modules=OutputModuleContainer(torch.nn.ModuleDict({
            'bar1': DummyRecurrentOutputModule(
                **common_params,
                num_heads=num_heads,
                hidden_size=[(7, 8, 9), (10, 1)],
            ),
            'bar2': DummyRecurrentOutputModule(
                **common_params,
                num_heads=num_heads,
                hidden_size=[(2,), (3,), (4,)],
            ),
        })),
        key_size=key_size,
        value_size=value_size,
        num_heads=num_heads,
    )

    hidden = model.init_hidden(3)
    print('Iteration 1')
    x = model({}, hidden)
    hidden = x['hidden']
    print('Iteration 2')
    x = model({}, hidden)
