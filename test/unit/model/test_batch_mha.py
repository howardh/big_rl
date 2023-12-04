import pytest

import torch

from big_rl.model.model import BatchMultiHeadAttentionEinsum, BatchMultiHeadAttentionBroadcast, BatchMultiHeadAttentionVmap, NonBatchMultiHeadAttention


@pytest.mark.parametrize("batch_mha_cls", [BatchMultiHeadAttentionEinsum, BatchMultiHeadAttentionBroadcast, BatchMultiHeadAttentionVmap])
def test_same_as_non_batch(batch_mha_cls):
    modules = [torch.nn.MultiheadAttention(embed_dim=6, num_heads=3) for _ in range(5)]
    batch_mha = batch_mha_cls(modules, num_heads=3, key_size=6, default_batch=True)
    non_batch_mha = NonBatchMultiHeadAttention(modules, num_heads=3, key_size=6, default_batch=True)

    seq_len = 7
    batch_size = 11
    query = torch.randn(5, batch_size, 6)
    key = torch.randn(5, seq_len, batch_size, 6)
    value = torch.randn(5, seq_len, batch_size, 6)

    batch_mha_output = batch_mha(query, key, value)
    non_batch_mha_output = non_batch_mha(query, key, value)

    assert torch.allclose(batch_mha_output[0], non_batch_mha_output[0], atol=1e-7)
    assert torch.allclose(batch_mha_output[1], non_batch_mha_output[1], atol=1e-7)


@pytest.mark.parametrize("batch_mha_cls", [BatchMultiHeadAttentionEinsum, BatchMultiHeadAttentionBroadcast, BatchMultiHeadAttentionVmap])
def test_convert_to_mha(batch_mha_cls):
    """ BatchMultiHeadAttentionEinsum/Broadcast can be converted back to a list of torch.nn.MultiheadAttention modules. Check that the conversion is done correctly. """
    modules = [torch.nn.MultiheadAttention(embed_dim=6, num_heads=3) for _ in range(5)]
    batch_mha = batch_mha_cls(modules, num_heads=3, key_size=6, default_batch=True)
    non_batch_mha = NonBatchMultiHeadAttention(batch_mha.to_multihead_attention_modules(), num_heads=3, key_size=6, default_batch=True)

    seq_len = 7
    batch_size = 11
    query = torch.randn(5, batch_size, 6)
    key = torch.randn(5, seq_len, batch_size, 6)
    value = torch.randn(5, seq_len, batch_size, 6)

    batch_mha_output = batch_mha(query, key, value)
    non_batch_mha_output = non_batch_mha(query, key, value)

    assert torch.allclose(batch_mha_output[0], non_batch_mha_output[0], atol=1e-7)
    assert torch.allclose(batch_mha_output[1], non_batch_mha_output[1], atol=1e-7)


@pytest.mark.parametrize("batch_mha_cls", [BatchMultiHeadAttentionEinsum, BatchMultiHeadAttentionBroadcast, BatchMultiHeadAttentionVmap])
def test_convert_to_mha_parameters_unchanged(batch_mha_cls):
    """  """
    modules = [torch.nn.MultiheadAttention(embed_dim=6, num_heads=3) for _ in range(5)]
    batch_mha1 = batch_mha_cls(modules, num_heads=3, key_size=6, default_batch=True)
    batch_mha2 = batch_mha_cls(batch_mha1.to_multihead_attention_modules(), num_heads=3, key_size=6, default_batch=True)

    for p1, p2 in zip(batch_mha1.parameters(), batch_mha2.parameters()):
        assert p1.shape == p2.shape
        assert (p1 == p2).all()
