import pytest
from big_rl.model.core_module.container import CoreModule
import torch

from big_rl.model.factory import create_core_modules
from big_rl.model.core_module import AVAILABLE_CORE_MODULES


KEY_SIZE = 5
VALUE_SIZE = 5


@pytest.mark.parametrize('cls', AVAILABLE_CORE_MODULES)
def test_defaults(cls):
    batch_size = 2
    num_inputs = 3

    config = {
        'type': cls.__name__,
        'key_size': KEY_SIZE,
        'value_size': VALUE_SIZE,
        'num_heads': 1,
    }
    module = create_core_modules(config)

    key = torch.randn(num_inputs, batch_size, KEY_SIZE)
    value = torch.randn(num_inputs, batch_size, VALUE_SIZE)
    hidden = module.init_hidden(batch_size)

    output = module(key, value, hidden)
    assert len(output['hidden']) == module.n_hidden
    assert output['key'].shape[1] == batch_size
    assert output['value'].shape[1] == batch_size
    assert output['key'].shape[2] == KEY_SIZE
    assert output['value'].shape[2] == VALUE_SIZE

    for _ in range(3):
        hidden = output['hidden']
        output = module(key, value, hidden)


@pytest.mark.parametrize('cls', AVAILABLE_CORE_MODULES)
def test_parallel(cls):
    """ Check that parameters (key_size, value_size, num_heads) are properly passed to the module """
    batch_size = 2
    num_inputs = 3

    config = {
        'container': 'parallel',
        'key_size': KEY_SIZE,
        'value_size': VALUE_SIZE,
        'num_heads': 1,
        'modules': [{
            'type': cls.__name__,
        }]
    }
    module = create_core_modules(config)

    key = torch.randn(num_inputs, batch_size, KEY_SIZE)
    value = torch.randn(num_inputs, batch_size, VALUE_SIZE)
    hidden = module.init_hidden(batch_size)

    output = module(key, value, hidden)
    assert len(output['hidden']) == module.n_hidden
    assert output['key'].shape[1] == batch_size
    assert output['value'].shape[1] == batch_size
    assert output['key'].shape[2] == KEY_SIZE
    assert output['value'].shape[2] == VALUE_SIZE

    for _ in range(3):
        hidden = output['hidden']
        output = module(key, value, hidden)
