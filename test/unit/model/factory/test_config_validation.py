import pytest

from pydantic import ValidationError

from big_rl.model.factory import create_model


def test_missing_input_modules():
    config = {
        'type': 'ModularModel1',
        #'input_modules': {},
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    with pytest.raises(ValidationError):
        create_model(config)


def test_missing_output_modules():
    config = {
        'type': 'ModularModel1',
        'input_modules': {},
        #'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    with pytest.raises(ValidationError):
        create_model(config)


def test_missing_core_modules():
    config = {
        'type': 'ModularModel1',
        'input_modules': {},
        'output_modules': {},
        #'core_modules': {
        #    'type': 'RecurrentAttention17',
        #},
    }

    with pytest.raises(ValidationError):
        create_model(config)


def test_negative_key_size():
    config = {
        'type': 'ModularModel1',
        'key_size': -1,
        'input_modules': {},
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    with pytest.raises(ValidationError):
        create_model(config)


def test_weight_config_invalid_filename():
    config = {
        'type': 'ModularModel1',
        'weight_config': {
            'filename': 'invalid_filename',
        },
        'input_modules': {},
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    with pytest.raises(ValidationError):
        create_model(config)


def test_core_module_specified_container_and_module_params():
    config = {
        'type': 'ModularModel1',
        'input_modules': {},
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
            'kwargs': {},
            'container': 'parallel',
        },
    }

    with pytest.raises(ValidationError):
        create_model(config)


def test_nonmodular_models():
    """ Non-modular models configs are valid without input/output/core modules """
    config = {
        'type': 'LSTMModel1',
        'kwargs': {
            'input_size': 7,
            'hidden_size': 5,
            'action_size': 3,
        },
    }

    create_model(config)
