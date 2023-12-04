import pytest

import torch

from big_rl.model.core_module.recurrent_attention_17 import RecurrentAttention17, NonBatchRecurrentAttention17
from big_rl.model.core_module.clock import ClockCoreModule
from big_rl.model.core_module.generalized_hebbian_algorithm import BatchAttentionGHA, AttentionGHA


@pytest.mark.parametrize('cls', [RecurrentAttention17, NonBatchRecurrentAttention17, BatchAttentionGHA, AttentionGHA, ClockCoreModule])
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
