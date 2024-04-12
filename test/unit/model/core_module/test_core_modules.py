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


@pytest.mark.parametrize('cls', AVAILABLE_CORE_MODULES)
def test_batch_dimensions_correct(cls):
    """ Make sure that the batch dimension is handled correctly. We check this by making sure that when we change the data in one batch, it does not affect that of another batch. """
    input_size = 4
    batch_size = 4
    seq_len = 8

    if cls.__name__ == 'ClockCoreModule':
        return # Too complicated to test. The hidden state is different for each element in the batch, and it ignores the input so the batch with modified inputs don't actually produce different outputs.

    module = cls(key_size=input_size, value_size=input_size, num_heads=2)

    # Initialize hidden state.
    # Some core modules (e.g. ClockCoreModule) produce different hidden states for each element in the batch. We need to modify the hidden state so that it's the same across the batch.
    #hidden = module.init_hidden(batch_size=1)
    #hidden = tuple(
    #    torch.cat([h] * batch_size, dim=d)
    #    for h,d in zip(hidden, module.hidden_batch_dims)
    #)
    hidden = module.init_hidden(batch_size=batch_size)

    k = torch.rand(seq_len, 1, input_size).repeat(1, batch_size, 1)
    v = torch.rand(seq_len, 1, input_size).repeat(1, batch_size, 1)

    output1 = module(k, v, hidden)

    if output1['key'].size(0) == 0:
        return # Some core modules (e.g. containers) don't return anything useful when initialized this way, so we skip them.

    k[:, 0, :] = 0
    v[:, 0, :] = 0

    output2 = module(k, v, hidden)

    assert torch.isclose(output2['value'][:,1:,:], output1['value'][:,1:,:]).all(), "Sanity check"
    assert torch.isclose(output2['key'][:,1:,:], output1['key'][:,1:,:]).all(), "Sanity check"

    # Changed batch may be different from the other batches.
    assert not torch.isclose(output2['value'][:, 0, :], output2['value'][:, 1, :]).all()
    # Unchanged batch should be the same.
    assert torch.isclose(output2['value'][:, 1, :], output2['value'][:, 2, :]).all()
    assert torch.isclose(output2['value'][:, 1, :], output2['value'][:, 3, :]).all()
