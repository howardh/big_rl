import itertools
from typing import List, Dict, Sequence, Tuple
import math

import torch
from torchtyping.tensor_type import TensorType
from torch.utils.data.dataloader import default_collate


# Recurrences

class RecurrentAttention(torch.nn.Module):
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(
                query, input_keys, input_values, average_attn_weights=True) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': attn_output.squeeze(0), # (batch_size, value_size)
        }


class RecurrentAttention2(torch.nn.Module):
    # Output to next block is computed from the attention output rather than just being the raw attention output
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention2, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size)
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': output_x # (batch_size, value_size)
        }


class RecurrentAttention3(torch.nn.Module):
    # Output to next block is gated
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention3, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size)
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, 1),
                torch.nn.Sigmoid()
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, 1)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'output_gate': output_gate.squeeze(1), # (batch_size)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*attn_output.squeeze(0) # (batch_size, value_size)
        }


class RecurrentAttention4(torch.nn.Module):
    # Bounded outputs with tanh
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention4, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size),
                torch.nn.Tanh(),
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size),
                torch.nn.Tanh(),
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size),
                torch.nn.Tanh(),
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size),
                torch.nn.Tanh(),
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, 1),
                torch.nn.Sigmoid()
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, 1)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*attn_output.squeeze(0) # (batch_size, value_size)
        }


class RecurrentAttention5(RecurrentAttention):
    # Just add tanh to RecurrentAttention, which we already know to work
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['x'] = output['x'].tanh()
        output['key'] = output['x'].tanh()
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention6(RecurrentAttention):
    # RecurrentAttention5 doesn't seem to work. Try only using tanh on the value output, since that's the part that propagates through time and has a higher potential of exploding.
    # This is working
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention7(RecurrentAttention2):
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention8(RecurrentAttention3):
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention9(RecurrentAttention3):
    # Same as RecurrentAttention8, but the feed-forward output gating chooses between something computed from the MHA output and the feed-forward input rather than between two things computed from the attention
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, 1)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'output_gate': output_gate.squeeze(1), # (batch_size)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0).tanh(), # (batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*x # (batch_size, value_size)
        }


class RecurrentAttention10(RecurrentAttention3):
    # Same as RecurrentAttention9, but the feed-forward output is held close to the initial input.
    # Note: this is incompatible with the other RecurrentAttension models, since the `forward` method takes an additional argument
    def forward(self,
            x: TensorType['num_blocks', 'batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float],
            initial_x: TensorType['batch_size','value_size',float],
        ):
        query = self.fc_query(x) # (num_blocks, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (num_blocks, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (num_blocks, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (num_blocks, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size)
        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights.permute(1,0,2), # (num_blocks, batch_size, seq_len)
            'output_gate': output_gate.squeeze(2), # (num_blocks, batch_size)
            'key': output_keys, # (num_blocks, batch_size, key_size)
            'value': output_values.tanh(), # (num_blocks, batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*initial_x # (num_blocks, batch_size, value_size)
        }


class RecurrentAttention11(torch.nn.Module):
    # Same as RecurrentAttention10, but removed the linear layers between the input and the query. The input is used as the query as is.
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention11, self).__init__()
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size)
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, 1),
                torch.nn.Sigmoid()
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['num_blocks', 'batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float],
            initial_x: TensorType['batch_size','value_size',float],
        ):
        query = x # (num_blocks, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (num_blocks, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (num_blocks, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (num_blocks, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size)
        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights.permute(1,0,2), # (num_blocks, batch_size, seq_len)
            'output_gate': output_gate.squeeze(2), # (num_blocks, batch_size)
            'key': output_keys, # (num_blocks, batch_size, key_size)
            'value': output_values.tanh(), # (num_blocks, batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*initial_x # (num_blocks, batch_size, value_size)
        }


class RecurrentAttention12(RecurrentAttention11):
    def forward(self,
            x: TensorType['num_blocks', 'batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float],
            initial_x: TensorType['batch_size','value_size',float],
        ):
        query = x # (num_blocks, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (num_blocks, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (num_blocks, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (num_blocks, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size)
        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights.permute(1,0,2), # (num_blocks, batch_size, seq_len)
            'output_gate': output_gate.squeeze(2), # (num_blocks, batch_size)
            'key': output_keys.tanh(), # (num_blocks, batch_size, key_size)
            'value': output_values.tanh(), # (num_blocks, batch_size, value_size)
            'x': output_gate*output_x.tanh() + (1-output_gate)*initial_x.tanh() # (num_blocks, batch_size, value_size)
        }


class RecurrentAttention13(RecurrentAttention3):
    """ Same as RecurrentAttention12, but without the weight sharing. This is meant to be used with ModularPolicy4.
    Added some ReLUs as well so we don't have two consecutive linear layers.
    """
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention3, self).__init__()
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size),
                torch.nn.ReLU(),
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size),
                torch.nn.ReLU(),
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size),
                torch.nn.ReLU(),
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, 1),
                torch.nn.Sigmoid(),
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
        self.initial_hidden_state = torch.nn.Parameter(
                torch.rand([input_size], requires_grad=True))
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float],
        ):
        #initial_x: TensorType['batch_size', 'value_size',float] = self.initial_hidden_state.expand(x.size(0), -1)
        query = x.unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size)
        return { # seq_len = number of inputs receives
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'output_gate': output_gate.squeeze(1), # (batch_size)
            'key': output_keys.squeeze(0).tanh(), # (batch_size, key_size)
            'value': output_values.squeeze(0).tanh(), # (batch_size, value_size)
            'x': output_gate*output_x.tanh() + (1-output_gate)*self.initial_hidden_state.tanh() # (batch_size, value_size)
        }


class RecurrentAttention14(RecurrentAttention11):
    """ """
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size, num_modules):
        super(RecurrentAttention11, self).__init__()
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(input_size, ff_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(ff_size, key_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(input_size, ff_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(ff_size, value_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(input_size*2, ff_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(ff_size, input_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(input_size*2, ff_size) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.ReLU(),
                BatchLinear([
                    torch.nn.Linear(ff_size, 1) for _ in range(num_modules)
                ], default_batch=True),
                torch.nn.Sigmoid(),
        )
        self.attention = BatchMultiHeadAttentionEinsum([
            torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
            for _ in range(num_modules)
        ], key_size=key_size, num_heads=num_heads, default_batch=True)
        #self.attention = NonBatchMultiHeadAttention([
        #    torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
        #    for _ in range(num_modules)
        #], key_size=key_size, num_heads=num_heads, default_batch=True)
    def forward(self,
            x: TensorType['num_blocks','batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float],
            initial_x: TensorType['num_blocks','input_size',float],
        ):
        num_modules = x.size(0)
        query = x # (num_blocks, batch_size, input_size)
        attn_output, attn_output_weights = self.attention(
                query, 
                input_keys.expand([num_modules, *input_keys.shape]),
                input_values.expand([num_modules, *input_values.shape])
        ) # (num_blocks, 1, batch_size, value_size), (num_modules, batch_size, 1, seq_len) -- Extra size 1 dimension is the number of queries. We only provide 1 query per module, so it's size 1.
        attn_output = attn_output.squeeze(1) # (num_blocks, batch_size, value_size)
        attn_output_weights = attn_output_weights.squeeze(2) # (num_blocks, batch_size, seq_len)
        output_keys = self.fc_key(attn_output) # (num_blocks, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (num_blocks, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output,x], dim=2)) # (num_blocks, batch_size)
        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights, # (num_blocks, batch_size, seq_len)
            'output_gate': output_gate.squeeze(2), # (num_blocks, batch_size)
            'key': output_keys.tanh(), # (num_blocks, batch_size, key_size)
            'value': output_values.tanh(), # (num_blocks, batch_size, value_size)
            'x': output_gate*output_x.tanh() + (1-output_gate)*initial_x.tanh() # (num_blocks, batch_size, value_size)
        }


class RecurrentAttention15(RecurrentAttention11):
    """ Same as RecurrentAttention14, but added gating to the output keys and values. 
    
    API changes:
    - `forward()` takes three inputs: state, key, value. It also outputs a dictionary with the same three keys.
        - Debugging values: `attn_output`, `attn_output_weights`, `gates`
    - `init_hidden(batch_size)` returns a tuple which is used to initialize the state.

    Previously, `forward()` had a `initial_x` input which was used as a default query. This doesn't change between batches, so it makes more sense for this to be a parameter of the model than an input. Unclear why I made it a parameter of the parent module rather than of the recurrence module.
    """
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size, num_modules, batch_type: str = 'einsum'):
        super(RecurrentAttention11, self).__init__()

        # Save parameters
        self._input_size = input_size
        self._num_modules = num_modules

        # Initialize fully-connected modules
        self.fc_query = self._make_mlp([input_size*2, ff_size, key_size])
        self.fc_key = self._make_mlp([input_size*2, ff_size, key_size])
        self.fc_value = self._make_mlp([input_size*2, ff_size, value_size])
        self.fc_state = self._make_mlp([input_size*2, ff_size, input_size])

        self.fc_query_gate = self._make_mlp(
                [input_size*2, ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_key_gate = self._make_mlp(
                [input_size*2, ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_value_gate = self._make_mlp(
                [input_size*2, ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_state_gate = self._make_mlp(
                [input_size*2, ff_size, 1],
                torch.nn.Sigmoid()
        )

        # Initialize attention module
        batch_type_mapping = {
            'einsum': BatchMultiHeadAttentionEinsum,
            'none': NonBatchMultiHeadAttention,
            'broadcast': BatchMultiHeadAttentionBroadcast,
        }
        if batch_type not in batch_type_mapping:
            raise ValueError(f"Unknown batch_type {batch_type}. Valid values are: {', '.join(batch_type_mapping.keys())}.")
        MhaClass = batch_type_mapping[batch_type]
        self.attention = MhaClass([
            torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
            for _ in range(num_modules)
        ], key_size=key_size, num_heads=num_heads, default_batch=True)

        # Initialize default state
        self.default_state = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros([num_modules, input_size])),
                torch.nn.Parameter(torch.zeros([num_modules, key_size])),
                torch.nn.Parameter(torch.zeros([num_modules, key_size])),
                torch.nn.Parameter(torch.zeros([num_modules, value_size])),
        ])

    def forward(self,
            state: Tuple[
                TensorType['num_blocks','batch_size','input_size',float], # Internal state
                TensorType['num_blocks','batch_size','key_size',float], # Previous query
                TensorType['num_blocks','batch_size','key_size',float], # Previous key
                TensorType['num_blocks','batch_size','value_size',float], # Previous value
            ],
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
        ):
        num_modules = self._num_modules
        assert num_modules == state[0].size(0)
        assert len(state) == 4

        prev_internal_state = state[0] # (num_blocks, batch_size, input_size)
        prev_query = state[1] # (num_blocks, batch_size, input_size)
        prev_key = state[2] # (num_blocks, batch_size, key_size)
        prev_value = state[3] # (num_blocks, batch_size, value_size)

        # attn_output: (num_blocks, 1, batch_size, value_size)
        # attn_output_weights: (num_blocks, batch_size, 1, seq_len)
        # The extra size 1 dimension is the number of queries. We only provide 1 query per module, so it's size 1.
        attn_output, attn_output_weights = self.attention(
                prev_query, 
                key.expand([num_modules, *key.shape]),
                value.expand([num_modules, *value.shape])
        )

        # Remove the extra query dimension
        attn_output = attn_output.squeeze(1) # (num_blocks, batch_size, value_size)
        attn_output_weights = attn_output_weights.squeeze(2) # (num_blocks, batch_size, seq_len)

        fc_input = torch.cat([attn_output, prev_internal_state], dim=2)

        output_queries = self.fc_query(fc_input).tanh() # (num_blocks, batch_size, key_size)
        output_keys = self.fc_key(fc_input).tanh() # (num_blocks, batch_size, key_size)
        output_values = self.fc_value(fc_input).tanh() # (num_blocks, batch_size, value_size)
        output_state = self.fc_state(fc_input).tanh() # (num_blocks, batch_size, input_size)

        output_query_gate = self.fc_query_gate(fc_input) # (num_blocks, batch_size)
        output_key_gate = self.fc_key_gate(fc_input) # (num_blocks, batch_size)
        output_value_gate = self.fc_value_gate(fc_input) # (num_blocks, batch_size)
        output_state_gate = self.fc_state_gate(fc_input) # (num_blocks, batch_size)

        gated_output_queries = output_query_gate * output_queries + (1 - output_query_gate) * prev_query
        gated_output_keys = output_key_gate * output_keys + (1 - output_key_gate) * prev_key
        gated_output_values = output_value_gate * output_values + (1 - output_value_gate) * prev_value
        gated_output_state = output_state_gate * output_state + (1 - output_state_gate) * prev_internal_state

        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights, # (num_blocks, batch_size, seq_len)
            'gates': {
                'query': output_query_gate, # (num_blocks, batch_size)
                'key': output_key_gate, # (num_blocks, batch_size)
                'value': output_value_gate, # (num_blocks, batch_size)
                'state': output_state_gate, # (num_blocks, batch_size)
            },
            'key': gated_output_keys, # (num_blocks, batch_size, key_size)
            'value': gated_output_values, # (num_blocks, batch_size, value_size)
            'state': (
                gated_output_state, # (num_blocks, batch_size, input_size)
                gated_output_queries, # (num_blocks, batch_size, key_size)
                gated_output_keys, # (num_blocks, batch_size, key_size)
                gated_output_values, # (num_blocks, batch_size, value_size)
            )
        }

    def _make_mlp(self, sizes, last_activation: torch.nn.Module = torch.nn.ReLU()):
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.ReLU())
            layers.append(
                BatchLinear([
                    torch.nn.Linear(in_size, out_size) for _ in range(self._num_modules)
                ], default_batch=True),
            )
        return torch.nn.Sequential(*layers, last_activation)

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_state
        )


class LSTM(torch.nn.Module): # TODO
    """ LSTM recurrence following the same API as RecurrentAttention15. """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)

        self.default_state = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros([hidden_size])),
            torch.nn.Parameter(torch.zeros([hidden_size])),
        ])

        raise NotImplementedError()

    def forward(self,
            state: Tuple,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
        ):

        # I don't think this works

        state = state
        key = key
        value = value
        return {
            'key': ...,
            'value': ...,
            'state': ()
        }

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_state
        )


# Batched stuff

class NonBatchMultiHeadAttention(torch.nn.Module):
    def __init__(self, modules, key_size, num_heads, default_batch=False):
        super(NonBatchMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.default_batch = default_batch
        self.attentions = torch.nn.ModuleList(modules)

    def forward_module(self, query, key, value, attention):
        query = query.unsqueeze(0)

        num_heads = self.num_heads
        embed_dim = self.key_size
        head_dim = embed_dim // num_heads
        num_inputs, batch_size, _ = key.shape

        w_q, w_k, w_v = attention.in_proj_weight.chunk(3)
        b_q, b_k, b_v = attention.in_proj_bias.chunk(3)

        q = query @ w_q.transpose(-2,-1) + b_q
        k = key @ w_k.transpose(-2,-1) + b_k
        v = value @ w_v.transpose(-2,-1) + b_v

        tgt_len, bsz, embed_dim = query.shape
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        q = q / math.sqrt(head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = attn_output @ attention.out_proj.weight.transpose(-2,-1) + attention.out_proj.bias

        return attn_output, attn_output_weights.view(batch_size, num_heads, 1, num_inputs).mean(1)

    def forward(self, query, key, value, batched=None):
        if batched is None:
            batched = self.default_batch
        if batched:
            return self.forward_batch(query, key, value)
        else:
            return self.forward_unbatched(query, key, value)

    def forward_unbatched(self, query, key, value):
        return tuple(
            torch.stack(o)
            for o in zip(*[
                self.forward_module(query, key, value, attention)
                for attention in self.attentions
            ])
        )

    def forward_batch(self, query, key, value):
        return tuple(
            torch.stack(o)
            for o in zip(*[
                self.forward_module(q, k, v, a)
                for q, k, v, a in zip(query, key, value, self.attentions)
            ])
        )

    def to_multihead_attention_modules(self):
        return self.attentions


class BatchMultiHeadAttentionBroadcast(torch.nn.Module):
    def __init__(self, modules: List[torch.nn.MultiheadAttention], key_size, num_heads, default_batch=False):
        super(BatchMultiHeadAttentionBroadcast, self).__init__()

        self.num_heads = num_heads
        self.key_size = key_size
        self.default_batch = default_batch

        self.attentions = torch.nn.ModuleList(modules)

    def forward(self, query, key, value, batched=None):
        if batched is None:
            batched = self.default_batch
        if batched:
            return self.forward_batch(query, key, value)
        else:
            return self.forward_unbatched(query, key, value)

    def forward_unbatched(self, query, key, value):
        """ Feed the same inputs to all MHA modules """
        #nbmha = NonBatchMultiHeadAttention(self.attentions, self.key_size, self.num_heads)
        #y = nbmha(query, key, value, batched=False)

        num_heads = self.num_heads
        embed_dim = self.key_size
        head_dim = embed_dim // num_heads
        batch_size, embed_dim = query.shape
        num_inputs, _, _ = key.shape

        num_modules = len(self.attentions)

        w_q, w_k, w_v = default_collate([a.in_proj_weight.chunk(3) for a in self.attentions]) # type: ignore
        b_q, b_k, b_v = default_collate([a.in_proj_bias.chunk(3) for a in self.attentions]) # type: ignore

        q = query.unsqueeze(0) @ w_q.transpose(-2,-1) + b_q.unsqueeze(1)
        k = key.unsqueeze(0) @ w_k.unsqueeze(1).transpose(-2,-1) + b_k.unsqueeze(1).unsqueeze(2)
        v = value.unsqueeze(0) @ w_v.unsqueeze(1).transpose(-2,-1) + b_v.unsqueeze(1).unsqueeze(2)

        #(torch.cat([q for q,k,v in y], dim=0) - q).abs() < 1e-7
        #(torch.stack([k for q,k,v in y], dim=0) - k).abs() < 1e-7
        #(torch.stack([v for q,k,v in y], dim=0) - v).abs() < 1e-7

        q = q.contiguous().view(1, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)

        # (torch.cat([q for q,k,v in y], dim=0) - q).abs() < 1e-7

        # (torch.cat([q for q,k,v in y], dim=1) - q).abs() < 1e-7
        # pp (torch.cat([q for q,k,v in y], dim=1) - q[0,1,:]).abs() < 1e-7

        q = q / math.sqrt(head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)

        # (torch.cat(y,dim=0) - attn_output).abs() < 1e-6

        attn_output = attn_output.transpose(0, 1).contiguous().view(1, num_modules, batch_size, 1, embed_dim)
        # (torch.cat(y,dim=1) - attn_output.squeeze()).abs() < 1e-6
        out_proj_weight = default_collate([a.out_proj.weight for a in self.attentions]) # type: ignore
        out_proj_bias = default_collate([a.out_proj.bias for a in self.attentions]) # type: ignore
        attn_output = attn_output @ out_proj_weight.unsqueeze(1).unsqueeze(0).transpose(-2,-1) + out_proj_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        attn_output = attn_output.squeeze(3).transpose(0,1)

        return attn_output, attn_output_weights.view(num_modules, batch_size, num_heads, 1, num_inputs).mean(2)

    def forward_batch(self, query, key, value):
        num_heads = self.num_heads
        embed_dim = self.key_size
        head_dim = embed_dim // num_heads
        num_modules, batch_size, embed_dim = query.shape
        num_modules, num_inputs, batch_size, embed_dim = key.shape

        assert num_modules == len(self.attentions)

        w_q, w_k, w_v = default_collate([a.in_proj_weight.chunk(3) for a in self.attentions]) # type: ignore
        b_q, b_k, b_v = default_collate([a.in_proj_bias.chunk(3) for a in self.attentions]) # type: ignore

        q = query @ w_q.transpose(-2,-1) + b_q.unsqueeze(1)
        k = key @ w_k.unsqueeze(1).transpose(-2,-1) + b_k.unsqueeze(1).unsqueeze(2)
        v = value @ w_v.unsqueeze(1).transpose(-2,-1) + b_v.unsqueeze(1).unsqueeze(2)

        q = q.contiguous().view(1, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)

        # (torch.cat([q for q,k,v in y], dim=0) - q).abs() < 1e-7

        # (torch.cat([q for q,k,v in y], dim=1) - q).abs() < 1e-7
        # pp (torch.cat([q for q,k,v in y], dim=1) - q[0,1,:]).abs() < 1e-7

        q = q / math.sqrt(head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)

        # (torch.cat(y,dim=0) - attn_output).abs() < 1e-6

        attn_output = attn_output.transpose(0, 1).contiguous().view(1, num_modules, batch_size, 1, embed_dim)
        # (torch.cat(y,dim=1) - attn_output.squeeze()).abs() < 1e-6
        out_proj_weight = default_collate([a.out_proj.weight for a in self.attentions]) # type: ignore
        out_proj_bias = default_collate([a.out_proj.bias for a in self.attentions]) # type: ignore
        attn_output = attn_output @ out_proj_weight.unsqueeze(1).unsqueeze(0).transpose(-2,-1) + out_proj_bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        attn_output = attn_output.squeeze(3)

        return attn_output.transpose(0,1), attn_output_weights.view(num_modules, batch_size, num_heads, 1, num_inputs).mean(2)

    def to_multihead_attention_modules(self):
        return self.attentions


class BatchMultiHeadAttentionEinsum(torch.nn.Module):
    def __init__(self, modules: List[torch.nn.MultiheadAttention], key_size, num_heads, default_batch=False):
        super(BatchMultiHeadAttentionEinsum, self).__init__()

        self.num_modules = len(modules)
        self.num_heads = num_heads
        self.key_size = key_size
        self.default_batch = default_batch

        self.in_weight = torch.nn.ParameterList([
            torch.nn.Parameter(x.detach())
            for x in default_collate([a.in_proj_weight.chunk(3) for a in modules])
        ])
        self.in_bias = torch.nn.ParameterList([
            torch.nn.Parameter(x.detach())
            for x in default_collate([a.in_proj_bias.chunk(3) for a in modules])
        ])
        self.out_weight = torch.nn.Parameter(
            default_collate([a.out_proj.weight for a in modules]).detach()
        )
        self.out_bias = torch.nn.Parameter(
            default_collate([a.out_proj.bias for a in modules]).unsqueeze(1).detach()
        )

    def forward(self, query, key, value, batched=None):
        if batched is None:
            batched = self.default_batch
        if batched:
            return self.forward_batch(query, key, value)
        else:
            return self.forward_unbatched(query, key, value)

    def forward_unbatched(self, query, key, value):
        """ Feed the same inputs to all MHA modules """
        #nbmha = NonBatchMultiHeadAttention(self.attentions, self.key_size, self.num_heads)
        #y = nbmha(query, key, value, batched=False)

        num_heads = self.num_heads
        embed_dim = self.key_size
        head_dim = embed_dim // num_heads
        batch_size, embed_dim = query.shape
        num_inputs, _, _ = key.shape

        num_modules = self.num_modules

        w_q, w_k, w_v = self.in_weight
        b_q, b_k, b_v = self.in_bias

        q = query.unsqueeze(0) @ w_q.transpose(-2,-1) + b_q.unsqueeze(1)
        k = torch.einsum('ijk,lmk -> lijm', key, w_k) + b_k.unsqueeze(1).unsqueeze(2)
        v = torch.einsum('ijk,lmk -> lijm', value, w_v) + b_v.unsqueeze(1).unsqueeze(2)

        q = q.contiguous().view(1, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)

        q = q / math.sqrt(head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(num_modules, batch_size, embed_dim)
        attn_output = torch.einsum ('nbi,nji-> nbj', attn_output, self.out_weight) + self.out_bias

        attn_output = attn_output.unsqueeze(1)

        return attn_output, attn_output_weights.view(num_modules, batch_size, num_heads, 1, num_inputs).mean(2)

    def forward_batch(self, query, key, value):
        num_heads = self.num_heads
        embed_dim = self.key_size
        head_dim = embed_dim // num_heads
        num_modules, batch_size, embed_dim = query.shape
        num_modules, num_inputs, batch_size, embed_dim = key.shape

        assert num_modules == self.num_modules

        w_q, w_k, w_v = self.in_weight
        b_q, b_k, b_v = self.in_bias

        q = query @ w_q.transpose(-2,-1) + b_q.unsqueeze(1)
        k = torch.einsum('lijk,lmk -> lijm', key, w_k) + b_k.unsqueeze(1).unsqueeze(2)
        v = torch.einsum('lijk,lmk -> lijm', value, w_v) + b_v.unsqueeze(1).unsqueeze(2)

        q = q.contiguous().view(1, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.transpose(0,1).contiguous().view(num_inputs, num_modules * batch_size * num_heads, head_dim).transpose(0, 1)

        q = q / math.sqrt(head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(-2, -1))
        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(num_modules, batch_size, embed_dim)
        attn_output = torch.einsum ('nbi,nji-> nbj', attn_output, self.out_weight) + self.out_bias

        attn_output = attn_output.unsqueeze(1)

        return attn_output, attn_output_weights.view(num_modules, batch_size, num_heads, 1, num_inputs).mean(2)

    def to_multihead_attention_modules(self):
        num_modules = self.num_modules
        num_heads = self.num_heads
        embed_dim = self.key_size

        modules = [
                torch.nn.MultiheadAttention(embed_dim, num_heads)
                for _ in range(num_modules)
        ]
        for i, m in enumerate(modules):
            m.in_proj_weight = torch.nn.Parameter(
                torch.cat([ self.in_weight[0][i], self.in_weight[1][i], self.in_weight[2][i] ])
            )
            m.in_proj_bias = torch.nn.Parameter(
                torch.cat([ self.in_bias[0][i], self.in_bias[1][i], self.in_bias[2][i] ])
            )
            m.out_proj.weight = torch.nn.Parameter(
                self.out_weight[i].detach()
            )
            m.out_proj.bias = torch.nn.Parameter(
                self.out_bias[i].squeeze(0).detach()
            )

        return modules


class NonBatchLinear(torch.nn.Module):
    def __init__(self, modules, default_batch=False):
        super(NonBatchLinear, self).__init__()

        self.default_batch = default_batch
        self.linear = torch.nn.ModuleList(modules)

    def forward_module(self, x, module):
        y = module(x)

        w = module.weight
        b = module.bias
        output = x @ w.transpose(-2,-1) + b

        assert (output == y).all()

        return output

    def forward(self, x, batched=None):
        if batched is None:
            batched = self.default_batch
        if batched:
            return self.forward_batch(x)
        else:
            return self.forward_unbatched(x)
    
    def forward_unbatched(self, x):
        return torch.stack([
            self.forward_module(x, module)
            for module in self.linear
        ])

    def forward_batch(self, batched_x):
        return torch.stack([
            self.forward_module(x, module)
            for x,module in zip(batched_x,self.linear)
        ])

    def to_linear_modules(self):
        return self.linear


class BatchLinear(torch.nn.Module):
    def __init__(self, modules: List[torch.nn.Linear], default_batch=False):
        super(BatchLinear, self).__init__()

        self.default_batch = default_batch

        self.weight = torch.nn.Parameter(
            torch.stack([l.weight for l in modules]).permute(0,2,1).detach()
        )
        self.bias = torch.nn.Parameter(
            torch.stack([l.bias for l in modules]).unsqueeze(1).detach()
        )

    def forward(self, x, batched=None):
        if batched is None:
            batched = self.default_batch
        if batched:
            return self.forward_batch(x)
        else:
            return self.forward_unbatched(x)

    def forward_unbatched(self, x):
        output = x @ self.weight + self.bias

        return output

    def forward_batch(self, batched_x):
        output = batched_x @ self.weight + self.bias

        return output

    def to_linear_modules(self) -> List[torch.nn.Linear]:
        num_modules, input_size, output_size= self.weight.shape
        modules = [torch.nn.Linear(input_size, output_size) for _ in range(num_modules)]
        for module, w, b in zip(modules, self.weight.permute(0,2,1), self.bias):
            module.weight = torch.nn.Parameter(w)
            module.bias = torch.nn.Parameter(b.squeeze(0))
        return modules

# Everything else


class GreyscaleImageInput(torch.nn.Module):
    def __init__(self, key_size: int, value_size: int, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.Flatten(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.fc_key = torch.nn.Linear(in_features=512, out_features=key_size)
        self.fc_value = torch.nn.Linear(in_features=512, out_features=value_size)
    def forward(self, x: TensorType['batch_size','frame_stack','height','width',float]):
        x = self.conv(x * self.scale)
        return {
            'key': self.fc_key(x),
            'value': self.fc_value(x),
        }


class ImageInput56(torch.nn.Module):
    def __init__(self, key_size: int, value_size: int, in_channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=4,stride=1),
            torch.nn.Flatten(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.fc_key = torch.nn.Linear(in_features=512, out_features=key_size)
        self.fc_value = torch.nn.Linear(in_features=512, out_features=value_size)
    def forward(self, x: TensorType['batch_size','channels','height','width',float]):
        x = self.conv(x.float() * self.scale)
        return {
            'key': self.fc_key(x),
            'value': self.fc_value(x),
        }


class ScalarInput(torch.nn.Module):
    def __init__(self, key_size: int, value_size: int):
        super().__init__()
        self.value_size = value_size
        self.key = torch.nn.Parameter(torch.rand([key_size]))
    def forward(self, value: TensorType['batch_size',float]):
        batch_size = int(torch.tensor(value.shape).prod().item())
        batch_shape = value.shape
        assert len(batch_shape) == 2
        assert batch_shape[-1] == 1, 'Last dimension of input to ScalarInput has to be size 1.'
        return {
            'key': self.key.view(1,-1).expand(batch_size, -1).view(*batch_shape[:-1],-1),
            'value': value.view(-1,1).expand(batch_size,self.value_size).view(*batch_shape[:-1],-1)
        }


class LinearInput(torch.nn.Module):
    def __init__(self, input_size: int, key_size: int, value_size: int, shared_key: bool = False):
        """
        Args:
            input_size: The size of the input vector.
            key_size: The size of the key.
            value_size: The size of the value.
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self.ff_value = torch.nn.Linear(in_features=input_size, out_features=value_size)
        if shared_key:
            self.key = torch.nn.Parameter(torch.rand([key_size])-0.5)
        else:
            self.ff_key = torch.nn.Linear(in_features=input_size, out_features=key_size)
    def forward(self, x: TensorType['batch_size',float]):
        batch_size = x.shape[0]
        return {
            'key': self.ff_key(x) if not self._shared_key else self.key.expand(batch_size, -1),
            'value': self.ff_value(x)
        }


class DiscreteInput(torch.nn.Module):
    def __init__(self, input_size: int, key_size: int, value_size: int, shared_key: bool = False):
        """
        Args:
            input_size: Number of possible input values.
            key_size: The size of the key.
            value_size: The size of the value.
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self._input_size = input_size
        self.value = torch.nn.Parameter(torch.rand([input_size, value_size])-0.5)
        if shared_key:
            self.key = torch.nn.Parameter(torch.rand([key_size])-0.5)
        else:
            self.key = torch.nn.Parameter(torch.rand([input_size, key_size])-0.5)
    def forward(self, x: TensorType['batch_size',int]):
        batch_size = int(torch.tensor(x.shape).prod().item())
        batch_shape = x.shape
        assert len(batch_shape) == 1, 'Input to DiscreteInput has to be a 1D tensor.'
        x = x.long().flatten()
        return {
            'key': self.key.expand(batch_size, -1).view(*batch_shape, -1) if self._shared_key else self.key[x,:].view(*batch_shape, -1),
            'value': self.value[x,:].view(*batch_shape, -1)
        }


class MatrixInput(torch.nn.Module):
    def __init__(self, input_size: Sequence[int], key_size: int, value_size: int, num_heads: int = 1, shared_key: bool = False):
        """
        Args:
            input_size: The size of the input matrix.
            key_size: The size of the key.
            value_size: The size of the value.
            num_heads: ...
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self._key_size = key_size
        self._value_size = value_size

        self.left_value = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty([num_heads, input_size[0]])
            )
        )
        self.right_value = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty([input_size[1], value_size // num_heads])
            )
        )

        if shared_key:
            self.key = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty(key_size)
                )
            )
        else:
            self.left_key = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty([num_heads, input_size[0]])
                )
            )
            self.right_key = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(
                    torch.empty([input_size[1], key_size // num_heads])
                )
            )
    def forward(self, x: TensorType['batch_size','dim1','dim2',float]):
        batch_size = x.shape[0]
        if self._shared_key:
            key = self.key.expand(batch_size, -1)
        else:
            key = self.left_key @ x @ self.right_key
            key = key.view(batch_size, self._key_size)
        return {
            'key': key,
            'value': (self.left_value @ x @ self.right_value).view(batch_size, self._value_size)
        }


class ModularPolicy(torch.nn.Module):
    def __init__(self, inputs, num_actions, input_size, key_size, value_size, num_heads, ff_size, num_blocks=1,
            recurrence_type='RecurrentAttention'):
        super().__init__()
        self.key_size = key_size
        self.input_size = input_size
        self.value_size = value_size

        self.input_modules = self._init_input_modules(inputs)

        recurrenceClasses = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention,
                    RecurrentAttention2,
                    RecurrentAttention3,
                    RecurrentAttention4,
                    RecurrentAttention5,
                    RecurrentAttention6,
                    RecurrentAttention7,
                    RecurrentAttention8,
                    RecurrentAttention9,
                ]
        }
        recurrenceCls = None
        if recurrence_type in recurrenceClasses:
            recurrenceCls = recurrenceClasses[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))
        self.attention = torch.nn.ModuleList([
                recurrenceCls(input_size, key_size, value_size, num_heads, ff_size)
                for _ in range(num_blocks)
        ])

        self.fc_output = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(input_size, 512),
                torch.nn.LeakyReLU(),
        )
        self.v = torch.nn.Linear(in_features=512,out_features=1)
        self.pi = torch.nn.Linear(in_features=512,out_features=num_actions)

        self.last_attention = None # Store the attention for analysis purposes
        self.last_ff_gating = None

    def _init_input_modules(self, input_configs: Dict[str,Dict]):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ScalarInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            input_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = self.key_size,
                    value_size = self.value_size)
        return torch.nn.ModuleDict(input_modules)

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2
        batch_size = hidden[0].shape[1]
        device = next(self.parameters()).device

        self.last_attention = []
        self.last_ff_gating = []

        input_keys = []
        input_vals = []
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            y = self.input_modules[k](x)
            input_keys.append(y['key'].unsqueeze(0))
            input_vals.append(y['value'].unsqueeze(0))

        keys = torch.cat([
            *input_keys,
            hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1]
        ], dim=0)

        new_hidden = []
        x = torch.zeros([batch_size, self.input_size], device=device)
        for attention in self.attention:
            output = attention(x, keys, values)
            x = output['x']
            new_hidden.append((output['key'], output['value']))
            self.last_attention.append([h.cpu().detach() for h in output['attn_output_weights']])
            self.last_ff_gating.append(output['output_gate'].cpu().detach())
        x = self.fc_output(x)

        return {
            'value': self.v(x),
            'action': self.pi(x),
            'hidden': default_collate(new_hidden)
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([len(self.attention), batch_size, self.key_size], device=device), # Key
                torch.zeros([len(self.attention), batch_size, self.key_size], device=device), # Query
        )


class LinearOutput(torch.nn.Module):
    def __init__(self, output_size: int, key_size: int, num_heads: int):
        super().__init__()
        self.output_size = output_size

        self.query = torch.nn.Parameter((torch.rand([key_size])-0.5)*0.01)
        self.attention = torch.nn.MultiheadAttention(
                key_size, num_heads=num_heads, batch_first=False)
        self.ff = torch.nn.Linear(key_size, output_size)
    def forward(self,
            key: TensorType['num_blocks','batch_size','hidden_size',float],
            value: TensorType['num_blocks','batch_size','hidden_size',float],
            ) -> Dict[str,TensorType]:
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


class StateIndependentOutput(torch.nn.Module):
    def __init__(self, output_size: int, key_size: int, num_heads: int):
        super().__init__()
        key_size = key_size # Unused
        num_heads = num_heads # Unused

        self.output_size = output_size
        self.output = torch.nn.Parameter(torch.zeros([output_size]))

    def forward(self,
            key: TensorType['num_blocks','batch_size','hidden_size',float],
            value: TensorType['num_blocks','batch_size','hidden_size',float],
            ) -> Dict[str,torch.Tensor]:
        assert len(key.shape) == 3, f'Key shape must be [num_blocks,batch_size,hidden_size]. Got {key.shape}'
        assert len(value.shape) == 3, f'Value shape must be [num_blocks,batch_size,hidden_size]. Got {value.shape}'
        device = next(self.parameters()).device
        num_blocks = key.shape[0]
        batch_size = key.shape[1]
        return {
            'output': self.output.expand(batch_size, -1),
            'attn_output_weights': torch.zeros([1, batch_size, num_blocks], device=device),
        }


class ModularPolicy2(torch.nn.Module):
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, ff_size, num_blocks=1, recurrence_type='RecurrentAttention'):
        super().__init__()
        self.key_size = key_size
        self.input_size = input_size
        self.value_size = value_size

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

        recurrenceClasses = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention,
                    RecurrentAttention2,
                    RecurrentAttention3,
                    RecurrentAttention4,
                    RecurrentAttention5,
                    RecurrentAttention6,
                    RecurrentAttention7,
                    RecurrentAttention8,
                    RecurrentAttention9,
                ]
        }
        recurrenceCls = None
        if recurrence_type in recurrenceClasses:
            recurrenceCls = recurrenceClasses[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))
        self.attention = torch.nn.ModuleList([
                recurrenceCls(input_size, key_size, value_size, num_heads, ff_size)
                for _ in range(num_blocks)
        ])

        # Store the attention for analysis purposes
        self.last_attention = None
        self.last_ff_gating = None
        self.last_output_attention = None

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            input_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2
        batch_size = hidden[0].shape[1]
        device = next(self.parameters()).device

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = []

        # Compute input to core module
        input_keys = []
        input_vals = []
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            y = self.input_modules[k](x)
            input_keys.append(y['key'].unsqueeze(0))
            input_vals.append(y['value'].unsqueeze(0))

        keys = torch.cat([
            *input_keys,
            hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1]
        ], dim=0)

        # Core module computation
        new_hidden = []
        x = torch.zeros([batch_size, self.input_size], device=device)
        for attention in self.attention:
            output = attention(x, keys, values)
            x = output['x']
            new_hidden.append((output['key'], output['value']))
            self.last_attention.append([h.cpu().detach() for h in output['attn_output_weights']])
            self.last_ff_gating.append(output['output_gate'].cpu().detach())
        new_hidden = default_collate(new_hidden)

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_hidden[1]
        ], dim=0)

        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention.append([h.cpu().detach() for h in y['attn_output_weights']])

        return {
            **output,
            'hidden': new_hidden
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([len(self.attention), batch_size, self.key_size], device=device), # Key
                torch.zeros([len(self.attention), batch_size, self.key_size], device=device), # Query
        )


class ModularPolicy3(torch.nn.Module): # TODO
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, ff_size, chain_length=1, depth=1, width=1, recurrence_type='RecurrentAttention'):
        super().__init__()
        self._key_size = key_size
        self._input_size = input_size
        self._value_size = value_size

        self._chain_length = chain_length
        self._depth = depth
        self._width = width

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

        self.attention = self._init_core_modules(
                recurrence_type = recurrence_type,
                input_size = input_size,
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads,
                ff_size = ff_size,
                chain_length = chain_length,
                depth = depth,
                width = width,
        )

        # Store the attention for analysis purposes
        self.last_attention = None
        self.last_ff_gating = None
        self.last_output_attention = None

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            input_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def _init_core_modules(self, recurrence_type, input_size, key_size, value_size, num_heads, ff_size, chain_length, depth, width):
        recurrence_classes = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention,
                    RecurrentAttention2,
                    RecurrentAttention3,
                    RecurrentAttention4,
                    RecurrentAttention5,
                    RecurrentAttention6,
                    RecurrentAttention7,
                    RecurrentAttention8,
                    RecurrentAttention9,
                ]
        }

        cls = None
        if recurrence_type in recurrence_classes:
            cls = recurrence_classes[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))
        output = []
        for _ in range(width):
            output.append([])
            for _ in range(depth):
                col = torch.nn.ModuleList([
                        cls(input_size, key_size, value_size, num_heads, ff_size)
                        for _ in range(chain_length)
                ])
                output[-1].append(col)
            output[-1] = torch.nn.ModuleList(output[-1])
        output = torch.nn.ModuleList(output)
        return output

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2
        batch_size = hidden[0].shape[1]
        device = next(self.parameters()).device

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = []

        # Compute input to core module
        input_keys = []
        input_vals = []
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            y = self.input_modules[k](x)
            input_keys.append(y['key'].unsqueeze(0))
            input_vals.append(y['value'].unsqueeze(0))

        keys = torch.cat([
            *input_keys,
            hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1]
        ], dim=0)

        # Core module computation
        new_hidden = []
        x = torch.zeros([batch_size, self._input_size], device=device)
        for attention in self.attention:
            output = attention(x, keys, values)
            x = output['x']
            new_hidden.append((output['key'], output['value']))
            self.last_attention.append([h.cpu().detach() for h in output['attn_output_weights']])
            self.last_ff_gating.append(output['output_gate'].cpu().detach())
        new_hidden = default_collate(new_hidden)

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_hidden[1]
        ], dim=0)

        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention.append([h.cpu().detach() for h in y['attn_output_weights']])

        return {
            **output,
            'hidden': new_hidden
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([self._chain_length, batch_size, self._key_size], device=device), # Key
                torch.zeros([self._chain_length, batch_size, self._key_size], device=device), # Query
        )


class ModularPolicy4(torch.nn.Module):
    """
    Transformer architecture in the style of a fully connected feedforward network.
    """
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, ff_size, architecture, recurrence_type='RecurrentAttention'):
        super().__init__()
        self._key_size = key_size
        self._input_size = input_size
        self._value_size = value_size

        self._architecture = architecture

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

        self.attention = self._init_core_modules(
                recurrence_type = recurrence_type,
                input_size = input_size,
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads,
                ff_size = ff_size,
                architecture = architecture,
        )

        #self._cuda_streams = self._init_cuda_streams()

        # Store the attention for analysis purposes
        self.last_attention = None
        self.last_ff_gating = None
        self.last_output_attention = None

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            input_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def _init_core_modules(self, recurrence_type, input_size, key_size, value_size, num_heads, ff_size, architecture):
        recurrence_classes = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention,
                    RecurrentAttention2,
                    RecurrentAttention3,
                    RecurrentAttention4,
                    RecurrentAttention5,
                    RecurrentAttention6,
                    RecurrentAttention7,
                    RecurrentAttention8,
                    RecurrentAttention9,
                    RecurrentAttention13,
                ]
        }

        cls = None
        if recurrence_type in recurrence_classes:
            cls = recurrence_classes[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))

        output = torch.nn.ModuleList([
            torch.nn.ModuleList([
                    cls(input_size, key_size, value_size, num_heads, ff_size)
                    for _ in range(size)
            ])
            for size in architecture
        ])
        return output

    def _init_cuda_streams(self):
        if not torch.cuda.is_available():
            return None

        num_streams = max(
                len(self.input_modules),
                *self._architecture,
                len(self.output_modules),
        )

        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        return streams

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        return self.forward_cpu(inputs, hidden)
        #return self.forward_cuda(inputs, hidden)

    def forward_cpu(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2+len(self.attention)

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = []

        # Compute input to core module
        input_labels = []
        input_keys = []
        input_vals = []
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            y = self.input_modules[k](x)
            input_keys.append(y['key'].unsqueeze(0))
            input_vals.append(y['value'].unsqueeze(0))
            input_labels.append(k)

        # Core module computation
        keys = torch.cat([
            *input_keys,
            hidden[0],
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1],
        ], dim=0)

        new_keys = hidden[0]
        new_values = hidden[1]
        internal_state : List[TensorType] = hidden[2:]
        new_internal_state = []
        layer_output = None
        for attn_layer, in_state in zip(self.attention, internal_state):
            layer_output = [
                attn(s, keys, values)
                for s,attn in zip(in_state, attn_layer) # type: ignore (ModuleList is not iterable???)
            ]
            layer_output = default_collate(layer_output)
            new_internal_state.append(layer_output['x'])
            #if layer_output is not None: # Save output from last layer
            new_keys = layer_output['key']
            new_values = layer_output['value']
            keys = new_keys
            values = new_values
            self.last_attention.append([h.cpu().detach() for h in layer_output['attn_output_weights']])
            self.last_ff_gating.append(layer_output['output_gate'].cpu().detach())

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_keys
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_values
        ], dim=0)

        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention.append([h.cpu().detach() for h in y['attn_output_weights']])

        return {
            **output,
            'hidden': (new_keys, new_values, *new_internal_state),
            'misc': {
                'core_output': layer_output,
                'input_labels': input_labels,
            }
        }

    def forward_cuda(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2+len(self.attention)
        assert self._cuda_streams is not None

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = []

        # Compute input to core module
        input_keys = []
        input_vals = []
        streams = iter(self._cuda_streams) # type: ignore
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            s = next(streams)
            with torch.cuda.stream(s):
                y = self.input_modules[k](x)
                input_keys.append(y['key'].unsqueeze(0))
                input_vals.append(y['value'].unsqueeze(0))
        torch.cuda.synchronize()

        keys = torch.cat([
            *input_keys,
            hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1]
        ], dim=0)

        # Core module computation
        new_keys = hidden[0]
        new_values = hidden[1]
        internal_state : List[TensorType] = hidden[2:]
        new_internal_state = []
        layer_output = None
        for attn_layer, in_state in zip(self.attention, internal_state):
            layer_output = [] # type: ignore
            for state,attn,s in zip(in_state, attn_layer, self._cuda_streams): # type: ignore (ModuleList is not iterable???)
                with torch.cuda.stream(s):
                    layer_output.append(attn(state, keys, values))
            torch.cuda.synchronize()
            layer_output = default_collate(layer_output)
            new_internal_state.append(layer_output['x'])
        if layer_output is not None: # Save output from last layer
            new_keys = layer_output['key']
            new_values = layer_output['value']
            self.last_attention.append([h.cpu().detach() for h in layer_output['attn_output_weights']])
            self.last_ff_gating.append(layer_output['output_gate'].cpu().detach())

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_keys
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_values
        ], dim=0)

        for (k,v),s in zip(self.output_modules.items(), self._cuda_streams): # type: ignore
            with torch.cuda.stream(s):
                y = v(keys, values)
                output[k] = y['output']
                self.last_output_attention.append([h.cpu().detach() for h in y['attn_output_weights']])
        torch.cuda.synchronize()

        return {
            **output,
            'hidden': (new_keys, new_values, *new_internal_state)
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Key
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Query
                *[torch.zeros([size, batch_size, self._input_size], device=device) for size in self._architecture], # Internal State
        )


class ModularPolicy5(torch.nn.Module):
    """
    Transformer architecture in the style of a fully connected feedforward network, but all blocks in a layer share weights.
    """
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, ff_size, architecture, recurrence_type='RecurrentAttention'):
        super().__init__()
        self._key_size = key_size
        self._input_size = input_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._ff_size = ff_size
        self._input_modules_config = inputs

        self._architecture = architecture

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

        self.attention = self._init_core_modules(
                recurrence_type = recurrence_type,
                input_size = input_size,
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads,
                ff_size = ff_size,
                architecture = architecture,
        )
        self.initial_hidden_state = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros([size, input_size]))
            for size in architecture
        ])

        # Store the attention for analysis purposes
        self.last_attention = None # [num_layers, batch_size, num_heads, seq_len]
        self.last_ff_gating = None
        self.last_output_attention = None

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] is None:
                input_modules[k] = v['module']
            else:
                if v['type'] not in valid_modules:
                    raise NotImplementedError(f'Unknown output module type: {v["type"]}')
                input_modules[k] = valid_modules[v['type']](
                        **v.get('config', {}),
                        key_size = key_size,
                        value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                    StateIndependentOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def _init_core_modules(self, recurrence_type, input_size, key_size, value_size, num_heads, ff_size, architecture):
        recurrence_classes = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention10,
                    RecurrentAttention11,
                    RecurrentAttention14,
                ]
        }

        cls = None
        if recurrence_type in recurrence_classes:
            cls = recurrence_classes[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))

        try: # Some architectures need the number of modules to be specified
            output = torch.nn.ModuleList([
                cls(input_size, key_size, value_size, num_heads, ff_size, layer_size)
                for layer_size in architecture
            ])
        except:
            output = torch.nn.ModuleList([
                cls(input_size, key_size, value_size, num_heads, ff_size)
                for _ in architecture
            ])
        return output

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2+len(self.initial_hidden_state)

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = {}

        # Compute input to core module
        input_labels = []
        input_keys = []
        input_vals = []
        for k,module in self.input_modules.items():
            module_config = self._input_modules_config[k]
            if 'inputs' in module_config:
                input_mapping = module_config['inputs']
                module_inputs = {}
                for dest_key, src_key in input_mapping.items():
                    if src_key not in inputs:
                        module_inputs = None
                        break
                    module_inputs[dest_key] = inputs[src_key]
                if module_inputs is not None:
                    y = module(**module_inputs)
                    input_labels.append(k)
                    input_keys.append(y['key'].unsqueeze(0))
                    input_vals.append(y['value'].unsqueeze(0))
            else:
                if k not in inputs:
                    continue # Skip this input module if no data is provided
                module_inputs = inputs[k]
                y = module(module_inputs)
                input_labels.append(k)
                input_keys.append(y['key'].unsqueeze(0))
                input_vals.append(y['value'].unsqueeze(0))
        self.last_input_labels = input_labels

        #batch_size = inputs['reward'].shape[0]
        #if batch_size > 1:
        #    breakpoint()

        keys = torch.cat([
            *input_keys,
            hidden[0],
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1],
        ], dim=0)

        self.last_keys = keys
        self.last_values = values

        # Core module computation
        new_keys = hidden[0]
        new_values = hidden[1]
        internal_state : List[TensorType] = hidden[2:]
        new_internal_state = []
        layer_output = None
        for attn_layer, in_state, init_state in zip(self.attention, internal_state, self.initial_hidden_state):
            layer_output = attn_layer(
                    input_keys = keys,
                    input_values = values,
                    x = in_state,
                    initial_x = init_state.view(-1, 1, self._input_size),
            )
            new_internal_state.append(layer_output['x'])
            #if layer_output is not None: # Save output from last layer
            new_keys = layer_output['key']
            new_values = layer_output['value']
            keys = new_keys
            values = new_values
            self.last_attention.append(
                    layer_output['attn_output_weights'].cpu().detach())
            self.last_ff_gating.append(
                    layer_output['output_gate'].cpu().detach())

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_keys
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_values
        ], dim=0)

        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention[k] = y['attn_output_weights'].cpu().detach().squeeze(1) # (batch_size, seq_len)

        self.last_hidden = (new_keys, new_values, *new_internal_state)

        return {
            **output,
            'hidden': self.last_hidden,
            'misc': {
                'core_output': layer_output,
                'input_labels': input_labels,
            }
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Key
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Value
                *[x.view(-1, 1, self._input_size).expand(-1, batch_size, self._input_size) for x in self.initial_hidden_state], # Internal State
        )

    @property
    def has_attention(self):
        return True


class ModularPolicy5LSTM(torch.nn.Module):
    """
    Same as ModularPolicy5, but with LSTM instead of attention
    """
    def __init__(self, inputs, outputs, value_size, hidden_size):
        super().__init__()
        self._value_size = value_size
        self._hidden_size = hidden_size
        self._input_modules_config = inputs

        self.input_modules = self._init_input_modules(inputs,
                key_size=1, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=hidden_size, num_heads=1)

        self.lstm = torch.nn.LSTMCell(
                sum(m.get('config',{}).get('value_size', value_size) for m in inputs.values()), # m['config']['value_size'] is the size of the outputs of each input module. Sum them up to get the total size of the concatenated input to the LSTM.
                hidden_size,
        )
        self.initial_hidden_state = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros([hidden_size])),
            torch.nn.Parameter(torch.zeros([hidden_size])),

        ])

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] is None:
                input_modules[k] = v['module']
            else:
                if v['type'] not in valid_modules:
                    raise NotImplementedError(f'Unknown output module type: {v["type"]}')
                input_modules[k] = valid_modules[v['type']](**{
                        'key_size': key_size,
                        'value_size': value_size,
                        **v.get('config', {}),
                })
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['batch_size','hidden_size']]):
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(self.parameters()).device

        # Compute input to core module
        input_labels = []
        input_vals = []
        for k,module in sorted(self.input_modules.items()):
            module_config = self._input_modules_config[k]
            if 'inputs' in module_config:
                input_mapping = module_config['inputs']
                module_inputs = {}
                for dest_key, src_key in input_mapping.items():
                    if src_key not in inputs:
                        module_inputs = None
                        break
                    module_inputs[dest_key] = inputs[src_key]
                if module_inputs is not None:
                    y = module(**module_inputs)
                    input_labels.append(k)
                    input_vals.append(y['value'])
            else:
                if k not in inputs:
                    # No data is provided, so fill with 0
                    y = torch.zeros([batch_size, self._input_modules_config[k].get('config',{}).get('value_size',self._value_size)], device=device) # XXX: not tested.
                else:
                    module_inputs = inputs[k]
                    y = module(module_inputs)['value']
                input_labels.append(k)
                input_vals.append(y)
        self.last_input_labels = input_labels

        values = torch.cat([
            *input_vals,
        ], dim=1)

        self.last_values = values

        # Core module computation
        h_1, c_1 = self.lstm(values, hidden)

        # Compute output
        output = {}

        value = h_1.view(1, batch_size, -1)
        key = torch.zeros_like(value, device=device)

        for k,v in self.output_modules.items():
            y = v(key, value)
            output[k] = y['output']

        self.last_hidden = (h_1, c_1)

        return {
            **output,
            'hidden': self.last_hidden,
            'misc': {
                #'core_output': layer_output,
                #'input_labels': input_labels,
            }
        }

    def init_hidden(self, batch_size: int = 1):
        #device = next(self.parameters()).device
        return (
                self.initial_hidden_state[0].view(1, self._hidden_size).expand(batch_size, -1),
                self.initial_hidden_state[1].view(1, self._hidden_size).expand(batch_size, -1)
        )

    @property
    def has_attention(self):
        return False


class ModularPolicy6(ModularPolicy5):
    # Added a tanh to the core module hidden states
    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Key
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Query
                *[x.view(-1, 1, self._input_size).tanh().expand(-1, batch_size, self._input_size) for x in self.initial_hidden_state], # Internal State
        )


class ModularPolicy7(torch.nn.Module):
    """
    Copied ModularPolicy5 and made modifications to work with the API changes in RecurrentAttention15.
    """
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, ff_size, architecture, recurrence_type='RecurrentAttention15'):
        super().__init__()
        self._key_size = key_size
        self._input_size = input_size
        self._value_size = value_size
        self._input_modules_config = inputs

        self._architecture = architecture

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

        self.attention = self._init_core_modules(
                recurrence_type = recurrence_type,
                input_size = input_size,
                key_size = key_size,
                value_size = value_size,
                num_heads = num_heads,
                ff_size = ff_size,
                architecture = architecture,
        )

        ## Store the attention for analysis purposes
        #self.last_attention = None # [num_layers, batch_size, num_heads, seq_len]
        #self.last_ff_gating = None
        #self.last_output_attention = None

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ImageInput56,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                    MatrixInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] is None:
                input_modules[k] = v['module']
            else:
                if v['type'] not in valid_modules:
                    raise NotImplementedError(f'Unknown output module type: {v["type"]}')
                input_modules[k] = valid_modules[v['type']](
                        **v.get('config', {}),
                        key_size = key_size,
                        value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                    StateIndependentOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def _init_core_modules(self, recurrence_type, input_size, key_size, value_size, num_heads, ff_size, architecture):
        recurrence_classes = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention15,
                ]
        }

        cls = None
        if recurrence_type in recurrence_classes:
            cls = recurrence_classes[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))

        output = torch.nn.ModuleList([
            cls(input_size, key_size, value_size, num_heads, ff_size, layer_size)
            for layer_size in architecture
        ])
        return output

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2+sum(self._state_sizes)

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = {}

        # Compute input to core module
        input_labels = []
        input_keys = []
        input_vals = []
        for k,module in self.input_modules.items():
            module_config = self._input_modules_config[k]
            if 'inputs' in module_config:
                input_mapping = module_config['inputs']
                module_inputs = {}
                for dest_key, src_key in input_mapping.items():
                    if src_key not in inputs:
                        module_inputs = None
                        break
                    module_inputs[dest_key] = inputs[src_key]
                if module_inputs is not None:
                    y = module(**module_inputs)
                    input_labels.append(k)
                    input_keys.append(y['key'].unsqueeze(0))
                    input_vals.append(y['value'].unsqueeze(0))
            else:
                if k not in inputs:
                    continue # Skip this input module if no data is provided
                module_inputs = inputs[k]
                y = module(module_inputs)
                input_labels.append(k)
                input_keys.append(y['key'].unsqueeze(0))
                input_vals.append(y['value'].unsqueeze(0))
        self.last_input_labels = input_labels

        keys = torch.cat([
            *input_keys,
            hidden[0],
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1],
        ], dim=0)

        self.last_keys = keys
        self.last_values = values

        internal_state : List[TensorType] = []
        is_idx = 2
        for size in self._state_sizes:
            internal_state.append(hidden[is_idx:is_idx+size]) # type: ignore
            is_idx += size

        # Core module computation
        new_keys = hidden[0]
        new_values = hidden[1]
        new_state = []
        layer_output = None
        for attn_layer, state in zip(self.attention, internal_state):
            layer_output = attn_layer(
                    key = keys,
                    value = values,
                    state = state,
            )
            new_state += layer_output['state']
            #if layer_output is not None: # Save output from last layer
            new_keys = layer_output['key']
            new_values = layer_output['value']
            keys = new_keys
            values = new_values
            #self.last_attention.append(
            #        layer_output['attn_output_weights'].cpu().detach())
            #self.last_ff_gating.append(
            #        layer_output['output_gate'].cpu().detach())

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_keys
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_values
        ], dim=0)

        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention[k] = y['attn_output_weights'].cpu().detach().squeeze(1) # (batch_size, seq_len)

        self.last_hidden = (new_keys, new_values, *new_state)

        return {
            **output,
            'hidden': self.last_hidden,
            'misc': {
                'core_output': layer_output,
                'input_labels': input_labels,
            }
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        state = [
            attn.init_hidden(batch_size) # type: ignore
            for attn in self.attention
        ]
        self._state_sizes = [len(s) for s in state]
        return (
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Key
                torch.zeros([self._architecture[-1], batch_size, self._key_size], device=device), # Value
                #*[x.view(-1, 1, self._input_size).expand(-1, batch_size, self._input_size) for x in self.initial_hidden_state], # Internal State
                *itertools.chain.from_iterable(state),
        )


if __name__ == '__main__':
    def test_mp4():
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = ModularPolicy4(
                inputs = {
                    'obs (image)': {
                        'type': 'ImageInput56',
                        'config': {
                            'in_channels': 3,
                        },
                    },
                    'reward': {
                        'type': 'ScalarInput',
                    },
                    'action': {
                        'type': 'DiscreteInput',
                        'config': {
                            'input_size': 5,
                        },
                    },
                },
                outputs = {
                    'value': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 1,
                        }
                    },
                    'action': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 5
                        }
                    },
                },
                input_size=512,
                key_size=512,
                value_size=512,
                num_heads=8,
                ff_size=1024,
                recurrence_type='RecurrentAttention9',
                architecture=[3,3]
        ).to(device)

        rand_input = {
                'obs (image)': torch.randn(1,3,56,56, device=device),
                'reward': torch.randn(1, device=device),
                'action': torch.randint(0, 5, (1,), device=device),
        }

        hidden = model.init_hidden()
        output = model(rand_input, hidden)
        breakpoint()
        output = output

    def test_mp7():
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = ModularPolicy7(
                inputs = {
                    'obs (image)': {
                        'type': 'ImageInput56',
                        'config': {
                            'in_channels': 3,
                        },
                    },
                    'reward': {
                        'type': 'ScalarInput',
                    },
                    'action': {
                        'type': 'DiscreteInput',
                        'config': {
                            'input_size': 5,
                        },
                    },
                },
                outputs = {
                    'value': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 1,
                        }
                    },
                    'action': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 5
                        }
                    },
                },
                input_size=512,
                key_size=512,
                value_size=512,
                num_heads=8,
                ff_size=1024,
                recurrence_type='RecurrentAttention15',
                architecture=[3,3]
        ).to(device)

        rand_input = {
                'obs (image)': torch.randn(1,3,56,56, device=device),
                'reward': torch.randn([1,1], device=device),
                'action': torch.randint(0, 5, (1,), device=device),
        }

        hidden = model.init_hidden()
        output = model(rand_input, hidden)
        breakpoint()
        output = output

    def benchmark_mp4():
        import timeit

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = ModularPolicy4(
                inputs = {
                    'obs (image)': {
                        'type': 'ImageInput56',
                        'config': {
                            'in_channels': 3,
                        },
                    },
                    'reward': {
                        'type': 'ScalarInput',
                    },
                    'action': {
                        'type': 'DiscreteInput',
                        'config': {
                            'input_size': 5,
                        },
                    },
                },
                outputs = {
                    'value': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 1,
                        }
                    },
                    'action': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 5
                        }
                    },
                },
                input_size=512,
                key_size=512,
                value_size=512,
                num_heads=8,
                ff_size=1024,
                recurrence_type='RecurrentAttention9',
                architecture=[24]
        ).to(device)

        batch_size = 1024
        rand_input = {
                'obs (image)': torch.randn(batch_size,3,56,56, device=device),
                'reward': torch.randn(batch_size, device=device),
                'action': torch.randint(0, 5, (batch_size,), device=device),
        }

        hidden = model.init_hidden(batch_size=batch_size)

        def foo():
            model(rand_input, hidden)

        def foo2():
            output = model(rand_input, hidden)
            loss = output['value'].mean() + output['action'].mean() # type: ignore
            loss.backward()

        num_iterations = 1_000

        print('-'*80)
        print(f'{num_iterations} iterations of forward only')
        total_time = timeit.Timer(foo).timeit(number=num_iterations)
        print(f'{num_iterations} iterations took {total_time} seconds')
        print(f'{num_iterations / total_time} iterations per second')
        print(f'{total_time / num_iterations} seconds per iteration')

        print('-'*80)
        print(f'{num_iterations} iterations of forward and backward')
        total_time = timeit.Timer(foo2).timeit(number=num_iterations)
        print(f'{num_iterations} iterations took {total_time} seconds')
        print(f'{num_iterations / total_time} iterations per second')
        print(f'{total_time / num_iterations} seconds per iteration')

    def benchmark_mp5():
        import timeit

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        model = ModularPolicy5(
                inputs = {
                    'obs (image)': {
                        'type': 'ImageInput56',
                        'config': {
                            'in_channels': 3,
                        },
                    },
                    'reward': {
                        'type': 'ScalarInput',
                    },
                    'action': {
                        'type': 'DiscreteInput',
                        'config': {
                            'input_size': 5,
                        },
                    },
                },
                outputs = {
                    'value': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 1,
                        }
                    },
                    'action': {
                        'type': 'LinearOutput',
                        'config': {
                            'output_size': 5
                        }
                    },
                },
                input_size=512,
                key_size=512,
                value_size=512,
                num_heads=8,
                ff_size=1024,
                recurrence_type='RecurrentAttention14',
                architecture=[24,24]
        ).to(device)

        #batch_size = 1024
        batch_size = 128
        rand_input = {
                'obs (image)': torch.randn(batch_size,3,56,56, device=device),
                'reward': torch.randn([batch_size,1], device=device),
                'action': torch.randint(0, 5, (batch_size,), device=device),
        }

        hidden = model.init_hidden(batch_size=batch_size)

        def foo():
            model(rand_input, hidden)

        def foo2():
            output = model(rand_input, hidden)
            loss = output['value'].mean() + output['action'].mean() # type: ignore
            loss.backward()

        num_iterations = 1_000

        # Compile JIT code before timing
        foo()
        foo2()

        print('-'*80)
        print(f'{num_iterations} iterations of forward only')
        total_time = timeit.Timer(foo).timeit(number=num_iterations)
        print(f'{num_iterations} iterations took {total_time} seconds')
        print(f'{num_iterations / total_time} iterations per second')
        print(f'{total_time / num_iterations} seconds per iteration')

        print('-'*80)
        print(f'{num_iterations} iterations of forward and backward')
        total_time = timeit.Timer(foo2).timeit(number=num_iterations)
        print(f'{num_iterations} iterations took {total_time} seconds')
        print(f'{num_iterations / total_time} iterations per second')
        print(f'{total_time / num_iterations} seconds per iteration')

    def test_batching():
        # Multiple linear layers with the same inputs to each one
        #in_features = 4
        ##out_features = 5
        #num_modules = 3
        #num_inputs = 2

        #key_size = 16
        #num_heads = 4
        #batch_size = 5

        ## 1D input
        #linears = [torch.nn.Linear(in_features=in_features, out_features=out_features) for _ in range(num_modules)]
        #x = torch.rand([in_features])
        #y = torch.stack([l(x) for l in linears])

        #weight = torch.stack([l.weight for l in linears])
        #bias = torch.stack([l.bias for l in linears])
        #y2 = weight @ x + bias

        #print(y == y2)

        ## 2D input
        #linears = [torch.nn.Linear(in_features=in_features, out_features=out_features) for _ in range(num_modules)]
        #x = torch.rand([batch_size, in_features])
        #y = torch.stack([l(x) for l in linears])
        #bl = BatchLinear(in_features=in_features, out_features=out_features, num_modules=num_modules)
        #bl2 = BatchLinear(in_features=out_features, out_features=in_features, num_modules=num_modules)
        #bl2.forward_batch(bl(x))

        #weight = torch.stack([l.weight for l in linears]).permute(0,2,1)
        #bias = torch.stack([l.bias for l in linears]).unsqueeze(1)
        #y2 = x @ weight + bias

        #print(y == y2)

        # Attention #1
        # in_proj_weight, in_proj_bias
        # ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']
        #input_query = torch.rand([num_inputs, batch_size, key_size])
        #input_key = torch.rand([num_inputs, batch_size, key_size])
        #input_value = torch.rand([num_inputs, batch_size, key_size])

        #nbmha = NonBatchMultiHeadAttention(num_modules=num_modules, num_heads=num_heads, key_size=key_size)
        #nbmha(input_query, input_key, input_value)
        #bmha = BatchMultiHeadAttentionBroadcast(num_modules=num_modules, num_heads=num_heads, key_size=key_size)
        #bmha(input_query, input_key, input_value)

        ## Attention #2
        #x = torch.rand([batch_size, in_features])
        #bl_q = BatchLinear(in_features=in_features, out_features=key_size, num_modules=num_modules)
        #bl_k = BatchLinear(in_features=in_features, out_features=key_size, num_modules=num_modules)
        #bl_v = BatchLinear(in_features=in_features, out_features=key_size, num_modules=num_modules)
        ##nbmha = NonBatchMultiHeadAttention(num_modules=num_modules, num_heads=num_heads, key_size=key_size)
        ##nbmha.forward(input_query, input_key, input_value)
        #bmha = BatchMultiHeadAttentionBroadcast(num_modules=num_modules, num_heads=num_heads, key_size=key_size)
        #bmha.forward_batch(bl_q(x), bl_k(x), bl_v(x))
        pass

    #benchmark_mp4()
    #benchmark_mp5()
    #test_batching()
    #benchmark_functorch()
    #benchmark_functorch_mha()
    test_mp7()

