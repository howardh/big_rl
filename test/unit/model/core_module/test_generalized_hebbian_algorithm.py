import pytest

import torch

from big_rl.model.core_module.generalized_hebbian_algorithm import BatchAttentionGHA, AttentionGHA


def test_gradients():
    """
    Verify that gradients are computed for all parameters.
    Regression test for the log learning rate not updating.
    Cause for the bug: The weights (i.e. hidden state) were initialized to 0, so the GHA update rule never changed the weights and the learning rate did nothing.
    """
    input_size = 4
    batch_size = 4
    seq_len = 8

    module = BatchAttentionGHA(key_size=input_size, value_size=input_size, num_heads=2, ff_size=[])

    hidden = module.init_hidden(batch_size=batch_size)
    k = torch.rand(seq_len, batch_size, input_size)
    v = torch.rand(seq_len, batch_size, input_size)

    output = module(k, v, hidden)

    dummy_loss = output['value'].mean() + output['key'].mean()
    dummy_loss.backward()

    assert module.fc[1].weight.grad is not None
    assert module.fc[1].bias.grad is not None
    assert module.log_learning_rate.grad is not None
    assert module.log_learning_rate.grad.item() != 0.0
