import pytest

import torch

from big_rl.model.core_module import AVAILABLE_CORE_MODULES


@pytest.mark.parametrize('cls', AVAILABLE_CORE_MODULES)
def test_works_without_error(cls):
    """ Initialize the module and pass in random data. Verify that it doesn't error. """
    input_size = 4
    batch_size = 4
    seq_len = 8

    module = cls(key_size=input_size, value_size=input_size, num_heads=2)

    hidden = module.init_hidden(batch_size=batch_size)
    k = torch.rand(seq_len, batch_size, input_size)
    v = torch.rand(seq_len, batch_size, input_size)

    output = module(k, v, hidden)

    # Check that it returns the correct output shape by passing it back into the module.
    hidden = output['hidden']
    k = torch.cat([output['key'], k], dim=0)
    v = torch.cat([output['value'], v], dim=0)

    output = module(k, v, hidden)


@pytest.mark.parametrize('cls', AVAILABLE_CORE_MODULES)
def test_deterministic_hidden_state(cls):
    """
    Verify that the hidden state is a deterministic function.
    Regression test for the hidden state being different when training, leading to a large action log likelihood difference and a policy gradient loss of infinity.
    """
    input_size = 4
    batch_size = 4

    module = cls(key_size=input_size, value_size=input_size, num_heads=2)

    hidden1 = module.init_hidden(batch_size=batch_size)
    hidden2 = module.init_hidden(batch_size=batch_size)

    for h1,h2 in zip(hidden1, hidden2):
        assert (h1 == h2).all()
