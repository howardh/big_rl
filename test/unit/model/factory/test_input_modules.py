import pytest
import gymnasium
import numpy as np
import torch

from big_rl.model.factory import create_input_modules


KEY_SIZE = 5
VALUE_SIZE = 3
MODULE_NAME = 'foo'


@pytest.mark.parametrize('batch_size', [1, 2])
def test_GreyscaleImageInput(batch_size):
    config = {
        MODULE_NAME: {
            'type': 'GreyscaleImageInput',
            'kwargs': {
                'in_channels': 3,
            }
        }
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Verify that the module can run without errors
    output = module[MODULE_NAME](torch.zeros(batch_size, 3, 84, 84))

    assert 'key' in output
    assert 'value' in output

    assert output['key'].shape == (batch_size, KEY_SIZE)
    assert output['value'].shape == (batch_size, VALUE_SIZE)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_ScalarInput(batch_size):
    config = {
        MODULE_NAME: {
            'type': 'ScalarInput',
            'kwargs': {}
        }
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Verify that the module can run without errors
    output = module[MODULE_NAME](torch.zeros(batch_size, 1))

    assert 'key' in output
    assert 'value' in output

    assert output['key'].shape == (batch_size, KEY_SIZE)
    assert output['value'].shape == (batch_size, VALUE_SIZE)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_DiscreteInput(batch_size):
    config = {
        MODULE_NAME: {
            'type': 'DiscreteInput',
            'kwargs': {
                'input_size': 10,
            }
        }
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Verify that the module can run without errors
    output = module[MODULE_NAME](torch.zeros(batch_size))

    assert 'key' in output
    assert 'value' in output

    assert output['key'].shape == (batch_size, KEY_SIZE)
    assert output['value'].shape == (batch_size, VALUE_SIZE)


def test_DiscreteInput_check_boundaries():
    batch_size = 2
    config = {
        MODULE_NAME: {
            'type': 'DiscreteInput',
            'kwargs': {
                'input_size': 5,
            }
        }
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Input size is set to 5, so the input should be in the range [0, 4]
    # Verify that it works with inputs of 0 and 4
    module[MODULE_NAME](torch.ones(batch_size) * 0)
    module[MODULE_NAME](torch.ones(batch_size) * 4)

    # Verify that it fails with inputs of -1 and 5
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.ones(batch_size) * -1)
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.ones(batch_size) * 5)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('input_size', [1, 3])
def test_LinearInput(batch_size, input_size):
    config = {
        MODULE_NAME: {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': input_size,
            }
        }
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Verify that the module can run without errors
    output = module[MODULE_NAME](torch.zeros(batch_size, input_size))

    assert 'key' in output
    assert 'value' in output

    assert output['key'].shape == (batch_size, KEY_SIZE)
    assert output['value'].shape == (batch_size, VALUE_SIZE)


def test_LinearInput_input_size_boundary():
    batch_size = 1
    config = {
        MODULE_NAME: {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 5,
            }
        }
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Should raise an error if the input size is not 5
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.zeros(batch_size, 4))
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.zeros(batch_size, 6))


@pytest.mark.parametrize('batch_size', [1, 2])
def test_all_modules_together(batch_size):
    config = {
        'greyscale': {
            'type': 'GreyscaleImageInput',
            'kwargs': {
                'in_channels': 3,
            }
        },
        'scalar': {
            'type': 'ScalarInput',
            'kwargs': {}
        },
        'discrete': {
            'type': 'DiscreteInput',
            'kwargs': {
                'input_size': 10,
            }
        },
        'linear': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': 5,
            }
        },
    }
    module = create_input_modules(config, key_size=KEY_SIZE, value_size=VALUE_SIZE)

    # Verify that the module can run without errors
    module['greyscale'](torch.zeros(batch_size, 3, 84, 84))
    module['scalar'](torch.zeros(batch_size, 1))
    module['discrete'](torch.zeros(batch_size))
    module['linear'](torch.zeros(batch_size, 5))


@pytest.mark.parametrize('space_type', ['action_space', 'observation_space'])
def test_param_value_from_discrete_space(space_type):
    """ Test that the value of a parameter can be obtained from a discrete action/observation space """
    batch_size = 2
    config = {
        MODULE_NAME: {
            'type': 'DiscreteInput',
            'kwargs': {
                'input_size': {
                    'source': space_type,
                    'accessor': '.n',
                },
            }
        }
    }
    module = create_input_modules(
        config, key_size=KEY_SIZE, value_size=VALUE_SIZE,
        **{space_type: gymnasium.spaces.Discrete(5)}
    )

    # Input size is set to 5, so the input should be in the range [0, 4]
    # Verify that it works with inputs of 0 and 4
    module[MODULE_NAME](torch.ones(batch_size) * 0)
    module[MODULE_NAME](torch.ones(batch_size) * 4)

    # Verify that it fails with inputs of -1 and 5
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.ones(batch_size) * -1)
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.ones(batch_size) * 5)


@pytest.mark.parametrize('space_type', ['action_space', 'observation_space'])
def test_param_value_from_box_space(space_type):
    """ Test that the value of a parameter can be obtained from a discrete action/observation space """
    batch_size = 2
    config = {
        MODULE_NAME: {
            'type': 'DiscreteInput',
            'kwargs': {
                'input_size': {
                    'source': space_type,
                    'accessor': '.shape[0]',
                },
            }
        }
    }
    module = create_input_modules(
        config, key_size=KEY_SIZE, value_size=VALUE_SIZE,
        **{space_type: gymnasium.spaces.Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]))}
    )

    # Input size is set to 3, so the input should be in the range [0, 2]
    # Verify that it works with inputs of 0 and 4
    module[MODULE_NAME](torch.ones(batch_size) * 0)
    module[MODULE_NAME](torch.ones(batch_size) * 2)

    # Verify that it fails with inputs of -1 and 3
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.ones(batch_size) * -1)
    with pytest.raises(ValueError):
        module[MODULE_NAME](torch.ones(batch_size) * 3)


@pytest.mark.parametrize('space_type', ['action_space', 'observation_space'])
def test_param_value_from_dict_space(space_type):
    """ Test that the value of a parameter can be obtained from a discrete action/observation space """
    batch_size = 2
    config = {
        'greyscale': {
            'type': 'GreyscaleImageInput',
            'kwargs': {
                'in_channels': {
                    'source': space_type,
                    'accessor': '["greyscale"].shape[0]',
                }
            }
        },
        'scalar': {
            'type': 'ScalarInput',
            'kwargs': {}
        },
        'discrete': {
            'type': 'DiscreteInput',
            'kwargs': {
                'input_size': {
                    'source': space_type,
                    'accessor': '["discrete"].n',
                }
            }
        },
        'linear': {
            'type': 'LinearInput',
            'kwargs': {
                'input_size': {
                    'source': space_type,
                    'accessor': '["linear"].shape[0]',
                }
            }
        },
    }
    dict_space = gymnasium.spaces.Dict({
        'greyscale': gymnasium.spaces.Box(low=np.zeros([3, 84, 84]), high=np.ones([3, 84, 84]) * 255),
        'scalar': gymnasium.spaces.Box(low=np.array([0]), high=np.array([1])),
        'discrete': gymnasium.spaces.Discrete(5),
        'linear': gymnasium.spaces.Box(low=np.zeros([3]), high=np.ones([3])),
    })
    module = create_input_modules(
        config, key_size=KEY_SIZE, value_size=VALUE_SIZE,
        **{space_type: dict_space}
    )

    # Image input module
    # Expecting 3 channels
    module['greyscale'](torch.zeros([batch_size, 3, 84, 84]))
    # Verify that it fails with inputs with 2 or 4 channels
    with pytest.raises(Exception):
        module['greyscale'](torch.zeros([batch_size, 2, 84, 84]))
    with pytest.raises(Exception):
        module['greyscale'](torch.zeros([batch_size, 4, 84, 84]))

    # Discrete input module
    # Expecting values in the range [0, 4]
    module['discrete'](torch.zeros(batch_size))
    module['discrete'](torch.ones(batch_size) * 4)
    with pytest.raises(ValueError):
        module['discrete'](torch.ones(batch_size) * -1)
    with pytest.raises(ValueError):
        module['discrete'](torch.ones(batch_size) * 5)

    # Linear module
    # Input size is set to 3
    module['linear'](torch.zeros(batch_size, 3))
    # Verify that it fails with inputs of size 2 and 4
    with pytest.raises(ValueError):
        module['linear'](torch.zeros(batch_size, 2))
    with pytest.raises(ValueError):
        module['linear'](torch.zeros(batch_size, 4))
