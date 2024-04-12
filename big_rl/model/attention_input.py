from __future__ import annotations

import torch
from torchtyping.tensor_type import TensorType


class AttentionInput(torch.nn.Module):
    """
    A component to be used with modules that receive their inputs via attention. It takes a set of key/value pairs and outputs a single vector.
    """
    def __init__(self, key_size: int, value_size: int, num_heads: int, dynamic_query: bool = False):
        super().__init__()

        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._dynamic_query = dynamic_query

        self.query = torch.nn.Parameter(torch.randn(1, 1, key_size))
        if self._dynamic_query:
            self.attention_query = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
        ) -> dict:

        batch_size = key.size(1)
        output = {}

        # attn_output: (num_blocks, 1, batch_size, value_size)
        # attn_output_weights: (num_blocks, batch_size, 1, seq_len)
        # The extra size 1 dimension is the number of queries. We only provide 1 query per module, so it's size 1.
        if self._dynamic_query:
            attn_output, attn_output_weights_mod = self.attention_query(
                    query=self.query.expand(1, batch_size, self._key_size),
                    key=key,
                    value=value,
            )
            query = attn_output # (1, batch_size, key_size)
            # We assume that the key and value are the same size. This would not work otherwise.

            output['attn_output_weights_mod'] = attn_output_weights_mod.squeeze(1)
        else:
            query = self.query.expand(1, batch_size, self._key_size)

        attn_output, attn_output_weights = self.attention(
                query=query,
                key=key,
                value=value,
        )

        output['attn_output'] = attn_output.squeeze(0) # (1, batch_size, value_size)
        output['attn_output_weights'] = attn_output_weights.squeeze(1) # (batch_size, seq_len)

        return output
