import pytest

import torch

from big_rl.model.attention_input import AttentionInput


@pytest.mark.parametrize("dq", [True, False])
def test_no_error(dq: bool):
    """ Make sure that it runs without error """
    module = AttentionInput(key_size=32, value_size=32, num_heads=4, dynamic_query=dq)

    key = torch.rand(2, 3, 32)
    value = torch.rand(2, 3, 32)

    module(key, value)
