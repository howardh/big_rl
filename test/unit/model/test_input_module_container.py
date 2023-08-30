import pytest

import torch

from big_rl.model.factory import create_input_modules
from big_rl.model.input_module.container import InputModuleContainer


KEY_SIZE = 5
VALUE_SIZE = 3


def test_empty():
    """ An empty input module container should return an empty tensor. """
    input_modules = create_input_modules({}, KEY_SIZE, VALUE_SIZE)
    module_container = InputModuleContainer(input_modules)

    output = module_container({})

    assert 'key' in output
    assert 'value' in output

    assert len(output['key']) == 0
    assert len(output['value']) == 0


def test_remapped_input_same_output():
    """ Check that the output does not change when the input is remapped. """
    config = {
        'a': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
    }
    input_modules = create_input_modules(config, KEY_SIZE, VALUE_SIZE)
    module_container_1 = InputModuleContainer(
        input_modules,
    )
    module_container_2 = InputModuleContainer(
        input_modules,
        input_mapping=[
            ('b', 'a'),
        ]
    )

    output_1 = module_container_1({'a': torch.zeros(1, 4)})
    output_2 = module_container_2({'b': torch.zeros(1, 4)})

    for a, b in zip(output_1['key'], output_2['key']):
        assert torch.all(torch.eq(a, b))
    for a, b in zip(output_1['value'], output_2['value']):
        assert torch.all(torch.eq(a, b))


def test_map_to_multiple_modules():
    """ Check that a single input can be mapped to multiple modules. """
    config = {
        'a': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
        'b': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
    }
    input_modules = create_input_modules(config, KEY_SIZE, VALUE_SIZE)
    module_container = InputModuleContainer(
        input_modules,
        input_mapping=[
            ('x', 'a'),
            ('x', 'b'),
        ]
    )

    output = module_container({'x': torch.zeros(1, 4)})

    assert len(output['key']) == 2
    assert len(output['value']) == 2


def test_mixed_remap_and_default_mapping():
    """ If we remap some inputs but not others, then the default mapping should be used for the unmapped inputs.
    """
    config = {
        'a': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
        'b': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
    }
    input_modules = create_input_modules(config, KEY_SIZE, VALUE_SIZE)
    module_container = InputModuleContainer(
        input_modules,
        input_mapping=[
            ('x', 'a'),
        ]
    )

    output = module_container({'x': torch.zeros(1, 4), 'b': torch.zeros(1, 4)})

    assert len(output['key']) == 2
    assert len(output['value']) == 2


def test_mixed_remap_and_default_mapping_2():
    """ If we remap some inputs but not others, then the default mapping should be used for the unmapped inputs.
    In this test, 'x' is mapped to both 'a' and 'b', producing 2 outputs, and 'b' has no mapping specified, so it is mapped to the 'b' module, producing one output. This gives us a total of 3 outputs.
    """
    config = {
        'a': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
        'b': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
    }
    input_modules = create_input_modules(config, KEY_SIZE, VALUE_SIZE)
    module_container = InputModuleContainer(
        input_modules,
        input_mapping=[
            ('x', 'a'),
            ('x', 'b'),
        ]
    )

    output = module_container({'x': torch.zeros(1, 4), 'b': torch.zeros(1, 4)})

    assert len(output['key']) == 3
    assert len(output['value']) == 3


def test_input_with_no_corresponding_module():
    """ If we pass in an input that has no corresponding module, it should error since it's very easy to have a silent bug where you think an input is being handled but it isn't. If an input module has no corresponding input, it should be ignored. """
    config = {
        'a': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
        'b': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
    }
    input_modules = create_input_modules(config, KEY_SIZE, VALUE_SIZE)
    module_container = InputModuleContainer(
        input_modules,
    )

    with pytest.raises(Exception):
        module_container({'non-existant-key': torch.zeros(1, 4)})


def test_invalid_module_name():
    """ If an input is mapped to a non-existent module, an error should be raised. """
    config = {
        'a': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
        'b': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 4,
            },
        },
    }
    input_modules = create_input_modules(config, KEY_SIZE, VALUE_SIZE)

    with pytest.raises(ValueError):
        InputModuleContainer(
            input_modules,
            input_mapping=[('x', 'c')]
        )
