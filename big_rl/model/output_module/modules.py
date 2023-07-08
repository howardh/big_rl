import torch

from jaxtyping import Float

from .base import OutputModule


class LinearOutput(OutputModule):
    def __init__(self, key_size: int, value_size: int, num_heads: int, output_size: int = 1):
        torch.nn.Module.__init__(self)

        if key_size != value_size:
            raise ValueError(f'`LinearOutput` does not support using different key and value sizes. Got {key_size} and {value_size}.')

        self.output_size = output_size

        self.query = torch.nn.Parameter((torch.rand([key_size])-0.5)*0.01)
        self.attention = torch.nn.MultiheadAttention(
                key_size, num_heads=num_heads, batch_first=False)
        self.ff = torch.nn.Linear(key_size, output_size)
    def forward(self,
            key: Float[torch.Tensor, 'num_blocks batch_size hidden_size'],
            value: Float[torch.Tensor, 'num_blocks batch_size hidden_size'],
            ) -> dict[str,torch.Tensor]:
        assert len(key.shape) == 3, f'Key shape must be [num_blocks,batch_size,hidden_size]. Got {key.shape}'
        assert len(value.shape) == 3, f'Value shape must be [num_blocks,batch_size,hidden_size]. Got {value.shape}'
        attn_output, attn_output_weights = self.attention(
                self.query.expand(1, key.shape[1], -1),
                key,
                value
        ) # (1, batch_size, value_size)
        output = self.ff(attn_output.squeeze(0))
        return {
            'output': output,
            'attn_output_weights': attn_output_weights,
        }


class StateIndependentOutput(OutputModule):
    def __init__(self, key_size: int, value_size: int, num_heads: int, output_size: int):
        torch.nn.Module.__init__(self)

        self.output_size = output_size
        self.output = torch.nn.Parameter(torch.zeros([output_size]))

    def forward(self,
            key: Float[torch.Tensor, 'num_blocks batch_size hidden_size'],
            value: Float[torch.Tensor, 'num_blocks batch_size hidden_size'],
            ) -> dict[str,torch.Tensor]:
        assert len(key.shape) == 3, f'Key shape must be [num_blocks,batch_size,hidden_size]. Got {key.shape}'
        assert len(value.shape) == 3, f'Value shape must be [num_blocks,batch_size,hidden_size]. Got {value.shape}'
        device = next(self.parameters()).device
        num_blocks = key.shape[0]
        batch_size = key.shape[1]
        return {
            'output': self.output.expand(batch_size, -1),
            'attn_output_weights': torch.zeros([1, batch_size, num_blocks], device=device),
        }
