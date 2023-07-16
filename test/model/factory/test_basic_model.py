import pytest

import torch
import gymnasium

from big_rl.model.factory import create_model

@pytest.mark.parametrize('batch_size', [1, 2])
def test_no_peripheral_modules(batch_size):
    """ Create a model with no input/output modules. Verify that it can run. """
    config = {
        'type': 'ModularModel1',
        'input_modules': {},
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }
    model = create_model(config)

    # Verify that the model can run without errors
    hidden = model.init_hidden(batch_size)
    for _ in range(10):
        output = model({}, hidden)
        hidden = output['hidden']


@pytest.mark.parametrize('batch_size', [1, 2])
def test_no_output_modules(batch_size):
    """ Create a model with input modules and no output modules. Verify that it can run. """
    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo': {
                'type': 'ScalarInput',
            }
        },
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }
    model = create_model(config)

    # Verify that the model can run without errors
    hidden = model.init_hidden(batch_size)
    for _ in range(10): # With inputs omitted
        output = model({}, hidden)
        hidden = output['hidden']
    for _ in range(10): # With inputs included
        output = model({'foo': torch.zeros(batch_size, 1)}, hidden)
        hidden = output['hidden']


@pytest.mark.parametrize('batch_size', [1, 2])
def test_full_model(batch_size):
    """ Create a model with input and output modules. """
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
    model = create_model(config)

    # Verify that the model can run without errors
    hidden = model.init_hidden(batch_size)
    for _ in range(10): # With inputs omitted
        output = model({}, hidden)
        hidden = output['hidden']
    for _ in range(10): # With inputs included
        output = model({'foo_scalar': torch.zeros(batch_size, 1)}, hidden)
        hidden = output['hidden']

    # It should error if it receives an input that isn't handled
    with pytest.raises(Exception):
        model({'this is an invalid key': torch.zeros(batch_size, 1)}, hidden)


@pytest.mark.parametrize('batch_size', [1])
def test_loading_input_weights(batch_size, tmpdir):
    """ """

    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_linear': {
                'type': 'LinearInput',
                'kwargs': {
                    'input_size': 3,
                },
            },
            'foo_discrete': {
                'type': 'DiscreteInput',
                'kwargs': {
                    'input_size': 3,
                },
            }
        },
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    model1 = create_model(config)
    model2 = create_model(config)

    filename = str(tmpdir.join('model.pt'))
    torch.save({'model': model1.state_dict()}, filename)

    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_linear': {
                'type': 'LinearInput',
                'kwargs': {
                    'input_size': 3,
                },
                'weight_config': {
                    'filename': filename,
                    'key_prefix': 'input_modules.input_modules.foo_linear',
                },
            },
        },
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    model3 = create_model(config)

    x = torch.zeros(batch_size, 3)

    y1 = model1.input_modules.input_modules['foo_linear'](x)
    y2 = model2.input_modules.input_modules['foo_linear'](x)
    y3 = model3.input_modules.input_modules['foo_linear'](x)

    # The two models created without specifying the weights should be different
    assert not torch.equal(y1['key'], y2['key'])
    assert not torch.equal(y1['value'], y2['value'])

    # Model3 should have the same weights as model1, and therefore should produce the same output
    assert torch.equal(y1['key'], y3['key'])
    assert torch.equal(y1['value'], y3['value'])


@pytest.mark.parametrize('batch_size', [1])
def test_loading_full_model_weights(batch_size, tmpdir):
    """ """

    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_linear': {
                'type': 'LinearInput',
                'kwargs': {
                    'input_size': 3,
                },
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
        },
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    model1 = create_model(config)
    model2 = create_model(config)

    filename = str(tmpdir.join('model.pt'))
    torch.save({'model': model1.state_dict()}, filename)

    config = {
        'weight_config': {
            'filename': filename,
            'key_prefix': '',
        },
        **config,
    }

    model3 = create_model(config)

    x = {
        'foo_linear': torch.zeros(batch_size, 3),
        'foo_discrete': torch.zeros(batch_size),
    }
    h = model1.init_hidden(batch_size)

    y1 = model1(x, h)
    y2 = model2(x, h)
    y3 = model3(x, h)

    # The two models created without specifying the weights should be different
    assert not torch.equal(y1['some_output'], y2['some_output'])

    # Model3 should have the same weights as model1, and therefore should produce the same output
    assert torch.equal(y1['some_output'], y3['some_output'])


@pytest.mark.parametrize('batch_size', [1])
def test_loading_full_model_weights_plus_submodule(batch_size, tmpdir):
    """ """

    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_linear': {
                'type': 'LinearInput',
                'kwargs': {
                    'input_size': 3,
                },
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
        },
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    model1 = create_model(config)
    model2 = create_model(config)

    filename = str(tmpdir.join('model.pt'))
    torch.save({'model': model1.state_dict()}, filename)

    config = {
        **config,
        'weight_config': {
            'filename': filename,
            'key_prefix': '',
        },
        'core_modules': {
            'type': 'RecurrentAttention17',
            'weight_config': {
                'filename': filename,
                'key_prefix': 'core_modules',
            }
        },
    }

    model3 = create_model(config)

    x = {
        'foo_linear': torch.zeros(batch_size, 3),
        'foo_discrete': torch.zeros(batch_size),
    }
    h = model1.init_hidden(batch_size)

    y1 = model1(x, h)
    y2 = model2(x, h)
    y3 = model3(x, h)

    # The two models created without specifying the weights should be different
    assert not torch.equal(y1['some_output'], y2['some_output'])

    # Model3 should have the same weights as model1, and therefore should produce the same output
    assert torch.equal(y1['some_output'], y3['some_output'])


@pytest.mark.parametrize('batch_size', [1])
def test_loading_full_model_weights_rearranged_core_modules(batch_size, tmpdir):
    """ """

    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'foo_linear': {
                'type': 'LinearInput',
                'kwargs': {
                    'input_size': 3,
                },
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
        },
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }

    model1 = create_model(config)
    model2 = create_model(config)

    filename = str(tmpdir.join('model.pt'))
    torch.save({'model': model1.state_dict()}, filename)

    config = {
        **config,
        'weight_config': {
            'filename': filename,
            'key_prefix': '',
        },
        'core_modules': {
            'container': 'parallel',
            'modules': [{
                'type': 'RecurrentAttention17',
                'weight_config': {
                    'filename': filename,
                    'key_prefix': 'core_modules',
                }
            }]
        },
    }

    model3 = create_model(config)

    x = {
        'foo_linear': torch.zeros(batch_size, 3),
        'foo_discrete': torch.zeros(batch_size),
    }
    h = model1.init_hidden(batch_size)

    y1 = model1(x, h)
    y2 = model2(x, h)
    y3 = model3(x, h)

    # The two models created without specifying the weights should be different
    assert not torch.equal(y1['some_output'], y2['some_output'])

    # Model3 should have the same weights as model1, and therefore should produce the same output
    assert torch.equal(y1['some_output'], y3['some_output'])


@pytest.mark.parametrize('batch_size', [1, 2])
def test_args_from_obs_or_action_space(batch_size):
    """ Check that the observation and action spaces are properly passed down to constituent modules and they can read their values for their kwargs. """
    config = {
        'type': 'ModularModel1',
        'input_modules': {
            'a': {
                'type': 'LinearInput',
                'kwargs': {
                    'input_size': {
                        'source': 'observation_space',
                        'accessor': '.shape[1]',
                    },
                }
            },
            'b': {
                'type': 'DiscreteInput',
                'kwargs': {
                    'input_size': {
                        'source': 'action_space',
                        'accessor': '.n',
                    },
                },
            }
        },
        'output_modules': {},
        'core_modules': {
            'type': 'RecurrentAttention17',
        },
    }
    model = create_model(
            config,
            observation_space=gymnasium.spaces.Box(low=-1, high=1, shape=(3,5,1)),
            action_space=gymnasium.spaces.Discrete(2),
    )

    # Verify that the model can run without errors
    hidden = model.init_hidden(batch_size)

    # 'a' expects a length 5 vector
    # 'b' expects a length 1 vector with values between 0 and 1
    model({
        'a': torch.zeros([batch_size, 5]),
        'b': torch.zeros([batch_size], dtype=torch.long)
    }, hidden)

    # It should error if 'a' is length 4 or 6
    with pytest.raises(Exception):
        model({
            'a': torch.zeros([batch_size, 4]),
            'b': torch.zeros([batch_size], dtype=torch.long)
        }, hidden)
    with pytest.raises(Exception):
        model({
            'a': torch.zeros([batch_size, 6]),
            'b': torch.zeros([batch_size], dtype=torch.long)
        }, hidden)

    # It should error if 'b' has values other than 0 or 1
    with pytest.raises(Exception):
        model({
            'a': torch.zeros([batch_size, 5]),
            'b': torch.zeros([batch_size], dtype=torch.long) + 2
        }, hidden)
    with pytest.raises(Exception):
        model({
            'a': torch.zeros([batch_size, 5]),
            'b': torch.zeros([batch_size], dtype=torch.long) - 1
        }, hidden)
