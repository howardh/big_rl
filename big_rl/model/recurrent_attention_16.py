import copy
import itertools
from typing import Tuple

import torch
from torchtyping.tensor_type import TensorType

from big_rl.model.model import BatchMultiHeadAttentionEinsum, NonBatchMultiHeadAttention, BatchMultiHeadAttentionBroadcast
from big_rl.model.model import BatchLinear, NonBatchLinear


class BatchRecurrentAttention16Layer_v2(torch.nn.Module):
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size, num_modules, batch_type: str = 'einsum'):
        super().__init__()

        # Preprocess and validate parameters
        assert input_size == key_size == value_size
        if isinstance(ff_size, int):
            ff_size = [ff_size]

        # Save parameters
        self._input_size = input_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._ff_size = ff_size
        self._num_modules = num_modules

        # Initialize fully-connected modules
        # TODO: These can be batched together. Instead of making 4 batches of `num_modules` linear layers, I can just make one batch of `4*num_modules` linear layers since they're all the same shape.
        #self.fc_query = self._make_mlp([input_size*2, *ff_size, key_size])
        #self.fc_key = self._make_mlp([input_size*2, *ff_size, key_size])
        #self.fc_value = self._make_mlp([input_size*2, *ff_size, value_size])
        #self.fc_state = self._make_mlp([input_size*2, *ff_size, input_size])

        self.fc_outputs = self._make_mlp(
                [input_size*2, *ff_size, value_size],
                num_duplicates=4,
        )
        self.fc_gates = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid(),
                num_duplicates=4,
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
        batch_size = key.shape[1]
        input_size = state[0].size(2)
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
        fc_input = fc_input.unsqueeze(0).expand(4, num_modules, batch_size, input_size*2).reshape(num_modules*4, batch_size, input_size*2)

        outputs = self.fc_outputs(fc_input).tanh() # (num_blocks*4, batch_size, key_size)
        #foo = outputs[:2]
        outputs = outputs.view([4, num_modules, batch_size, -1]) # (num_blocks, 4, batch_size, size)
        output_queries = outputs[0] # (num_blocks, batch_size, key_size)
        output_keys = outputs[1] # (num_blocks, batch_size, key_size)
        output_values = outputs[2] # (num_blocks, batch_size, value_size)
        output_state = outputs[3] # (num_blocks, batch_size, input_size)
        #breakpoint()

        gates = self.fc_gates(fc_input) # (num_blocks*4, batch_size)
        output_query_gate, output_key_gate, output_value_gate, output_state_gate = gates.view(4, num_modules, batch_size, 1)

        gated_output_queries = output_query_gate * output_queries + (1 - output_query_gate) * prev_query
        gated_output_keys = output_key_gate * output_keys + (1 - output_key_gate) * prev_key
        gated_output_values = output_value_gate * output_values + (1 - output_value_gate) * prev_value
        gated_output_state = output_state_gate * output_state + (1 - output_state_gate) * prev_internal_state

        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights, # (num_blocks, batch_size, seq_len)
            #'debug': outputs,
            #'outputs': {
            #    'query': output_queries,
            #    'key': output_keys,
            #    'value': output_values,
            #    'state': output_state,
            #},
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

    def _make_mlp(self, sizes, last_activation: torch.nn.Module = torch.nn.ReLU(), num_duplicates=1):
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.ReLU())
            layers.append(
                BatchLinear([
                    torch.nn.Linear(in_size, out_size) for _ in range(self._num_modules*num_duplicates)
                ], default_batch=True),
            )
        return torch.nn.Sequential(*layers, last_activation)

    def init_state(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_state
        )

    def to_nonbatched(self):
        return _batchedv2_to_nonbatched(self)

    @classmethod
    def from_nonbatched(cls, obj):
        return _to_batchedv2(obj)


class BatchRecurrentAttention16Layer(torch.nn.Module):
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size, num_modules, batch_type: str = 'einsum'):
        super().__init__()

        # Preprocess parameters
        if isinstance(ff_size, int):
            ff_size = [ff_size]

        # Save parameters
        self._input_size = input_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._ff_size = ff_size
        self._num_modules = num_modules

        # Initialize fully-connected modules
        # TODO: These can be batched together. Instead of making 4 batches of `num_modules` linear layers, I can just make one batch of `4*num_modules` linear layers since they're all the same shape.
        self.fc_query = self._make_mlp([input_size*2, *ff_size, key_size])
        self.fc_key = self._make_mlp([input_size*2, *ff_size, key_size])
        self.fc_value = self._make_mlp([input_size*2, *ff_size, value_size])
        self.fc_state = self._make_mlp([input_size*2, *ff_size, input_size])

        self.fc_query_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_key_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_value_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_state_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
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
            #'outputs': {
            #    'query': output_queries,
            #    'key': output_keys,
            #    'value': output_values,
            #    'state': output_state,
            #},
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

    def init_state(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_state
        )

    def to_nonbatched(self):
        return _to_nonbatched(self)

    @classmethod
    def from_nonbatched(cls, obj):
        return _to_batched(obj)


class NonBatchRecurrentAttention16Layer(torch.nn.Module):
    """ Same as RecurrentAttention14, but added gating to the output keys and values. 
    
    API changes:
    - `forward()` takes three inputs: state, key, value. It also outputs a dictionary with the same three keys.
        - Debugging values: `attn_output`, `attn_output_weights`, `gates`
    - `init_hidden(batch_size)` returns a tuple which is used to initialize the state.

    Previously, `forward()` had a `initial_x` input which was used as a default query. This doesn't change between batches, so it makes more sense for this to be a parameter of the model than an input. Unclear why I made it a parameter of the parent module rather than of the recurrence module.
    """
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size, num_modules):
        super().__init__()

        # Preprocess parameters
        if isinstance(ff_size, int):
            ff_size = [ff_size]

        # Save parameters
        self._input_size = input_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._ff_size = ff_size
        self._num_modules = num_modules

        # Initialize fully-connected modules
        self.fc_query = self._make_mlp([input_size*2, *ff_size, key_size])
        self.fc_key = self._make_mlp([input_size*2, *ff_size, key_size])
        self.fc_value = self._make_mlp([input_size*2, *ff_size, value_size])
        self.fc_state = self._make_mlp([input_size*2, *ff_size, input_size])

        self.fc_query_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_key_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_value_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )
        self.fc_state_gate = self._make_mlp(
                [input_size*2, *ff_size, 1],
                torch.nn.Sigmoid()
        )

        # Initialize attention module
        self.attention = NonBatchMultiHeadAttention([
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
            #'debug': (output_queries, output_keys, output_values, output_state),
            #'outputs': {
            #    'query': output_queries,
            #    'key': output_keys,
            #    'value': output_values,
            #    'state': output_state,
            #},
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
                NonBatchLinear([
                    torch.nn.Linear(in_size, out_size) for _ in range(self._num_modules)
                ], default_batch=True),
            )
        return torch.nn.Sequential(*layers, last_activation)

    def init_state(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_state
        )


class RecurrentAttention16(torch.nn.Module):
    """ Same as RecurrentAttention14, but added gating to the output keys and values. 
    
    API changes:
    - `forward()` takes three inputs: state, key, value. It also outputs a dictionary with the same three keys.
        - Debugging values: `attn_output`, `attn_output_weights`, `gates`
    - `init_hidden(batch_size)` returns a tuple which is used to initialize the state.

    Previously, `forward()` had a `initial_x` input which was used as a default query. This doesn't change between batches, so it makes more sense for this to be a parameter of the model than an input. Unclear why I made it a parameter of the parent module rather than of the recurrence module.
    """
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size, architecture=[]):
        super().__init__()

        self._architecture = architecture

        self._layers = torch.nn.ModuleList([
                BatchRecurrentAttention16Layer_v2(input_size, key_size, value_size, num_heads, ff_size, layer_size)
                for layer_size in architecture
        ])

    def forward(self,
            state: Tuple[torch.Tensor],
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
        ):
        # Split state into chunks of 4
        state_by_layer = [
            state[i:i+4] for i in range(0, len(state), 4)
        ]

        # Pass through each layer
        current_key = key
        current_value = value
        new_state = []
        for layer, layer_state in zip(self._layers, state_by_layer):
            layer_output = layer(layer_state, current_key, current_value)

            current_key = layer_output['key']
            current_value = layer_output['value']
            new_state.append(layer_output['state'])

        return {
            'key': current_key,
            'value': current_value,
            'state': tuple(itertools.chain(*new_state)),
        }

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        return tuple(itertools.chain.from_iterable(
            layer.init_state(batch_size) # type: ignore
            for layer in self._layers
        ))
    
    @property
    def num_outputs(self):
        """ Number of key-value pairs outputted by the model """
        return self._architecture[-1]

    @property
    def state_size(self):
        """ The number of elements in the state tuple. """
        return len(self._architecture) * 4


# Batching / Unbatching utils


def _to_nonbatched(rec: BatchRecurrentAttention16Layer) -> NonBatchRecurrentAttention16Layer:
    rec = copy.deepcopy(rec)
    output = NonBatchRecurrentAttention16Layer(
        input_size = rec._input_size,
        key_size = rec._key_size,
        value_size = rec._value_size,
        num_heads = rec._num_heads,
        ff_size = rec._ff_size,
        num_modules = rec._num_modules,
    )

    # Convert linear layers
    def convert(m):
        if not isinstance(m, BatchLinear):
            return m
        return NonBatchLinear(
                m.to_linear_modules(),
                default_batch=m.default_batch
        )
    linear_keys = 'fc_query', 'fc_key', 'fc_value', 'fc_state', 'fc_query_gate', 'fc_key_gate', 'fc_value_gate', 'fc_state_gate'
    for k in linear_keys:
        setattr(
                output, k,
                torch.nn.Sequential(
                    *[convert(x) for x in getattr(rec, k)]
                )
        )

    # Convert attention
    output.attention = NonBatchMultiHeadAttention(
        rec.attention.to_multihead_attention_modules(),
        key_size=rec._key_size,
        num_heads=rec._num_heads,
        default_batch=rec.attention.default_batch,
    )

    # Copy default state
    output.default_state = rec.default_state

    return output


def _batchedv2_to_nonbatched(rec: BatchRecurrentAttention16Layer_v2) -> NonBatchRecurrentAttention16Layer:
    rec = copy.deepcopy(rec)
    output = NonBatchRecurrentAttention16Layer(
        input_size = rec._input_size,
        key_size = rec._key_size,
        value_size = rec._value_size,
        num_heads = rec._num_heads,
        ff_size = rec._ff_size,
        num_modules = rec._num_modules,
    )

    #raise NotImplementedError()
    # Convert linear layers
    n = rec._num_modules
    def convert(sequential):
        query = []
        key = []
        value = []
        state = []
        for m in sequential:
            if not isinstance(m, BatchLinear):
                query.append(m)
                key.append(m)
                value.append(m)
                state.append(m)
                continue
            linear_modules = m.to_linear_modules()
            query.append(NonBatchLinear(
                    linear_modules[:n],
                    default_batch=m.default_batch
            ))
            key.append(NonBatchLinear(
                    linear_modules[n:2*n],
                    default_batch=m.default_batch
            ))
            value.append(NonBatchLinear(
                    linear_modules[2*n:3*n],
                    default_batch=m.default_batch
            ))
            state.append(NonBatchLinear(
                    linear_modules[3*n:],
                    default_batch=m.default_batch
            ))
        return torch.nn.Sequential(*query), torch.nn.Sequential(*key), torch.nn.Sequential(*value), torch.nn.Sequential(*state)

    output.fc_query, output.fc_key, output.fc_value, output.fc_state = convert(rec.fc_outputs)
    output.fc_query_gate, output.fc_key_gate, output.fc_value_gate, output.fc_state_gate = convert(rec.fc_gates)

    ## XXX: DEBUG
    #lin_modules = rec.fc_outputs[1].to_linear_modules()
    #in_features = lin_modules[0].in_features
    #b = 2
    #x = torch.randn(len(lin_modules), b, in_features)
    #x1 = torch.randn(len(lin_modules)//4, b, in_features)
    ##x1 = torch.arange(len(lin_modules)//4 * b * in_features).reshape(len(lin_modules)//4, b, in_features)
    #x2 = x1.unsqueeze(0).expand(4, n, b, in_features).reshape(n*4, b, in_features)
    ##x = torch.randn(len(lin_modules), b, in_features)
    ## rec.fc_outputs[1](x)[:2] == output.fc_query[1](x[:2])
    ## rec.fc_outputs(x)[:2] == output.fc_query(x[:2])
    ## rec.fc_outputs(x)[:2] == output.fc_query(x[:2])
    ## rec.fc_outputs(x).view(4,n,b,-1)[0] == output.fc_query(x[:2])
    ## rec.fc_outputs(x2).view(4,n,b,-1)[0] == output.fc_query(x1)
    #breakpoint()

    # Convert attention
    output.attention = NonBatchMultiHeadAttention(
        rec.attention.to_multihead_attention_modules(),
        key_size=rec._key_size,
        num_heads=rec._num_heads,
        default_batch=rec.attention.default_batch,
    )

    # Copy default state
    output.default_state = rec.default_state

    return output


def _to_batched(rec: NonBatchRecurrentAttention16Layer) -> BatchRecurrentAttention16Layer:
    rec = copy.deepcopy(rec)
    output = BatchRecurrentAttention16Layer(
        input_size = rec._input_size,
        key_size = rec._key_size,
        value_size = rec._value_size,
        num_heads = rec._num_heads,
        ff_size = rec._ff_size,
        num_modules = rec._num_modules,
    )

    # Convert linear layers
    def convert(m):
        if not isinstance(m, NonBatchLinear):
            return m
        return BatchLinear(
                m.to_linear_modules(),
                default_batch=m.default_batch
        )
    linear_keys = 'fc_query', 'fc_key', 'fc_value', 'fc_state', 'fc_query_gate', 'fc_key_gate', 'fc_value_gate', 'fc_state_gate'
    for k in linear_keys:
        setattr(
                output, k,
                torch.nn.Sequential(
                    *[convert(x) for x in getattr(rec, k)]
                )
        )

    # Convert attention
    output.attention = BatchMultiHeadAttentionEinsum(
        rec.attention.to_multihead_attention_modules(),
        key_size=rec._key_size,
        num_heads=rec._num_heads,
        default_batch=rec.attention.default_batch,
    )

    # Copy default state
    output.default_state = rec.default_state

    return output


def _to_batchedv2(rec: NonBatchRecurrentAttention16Layer) -> BatchRecurrentAttention16Layer_v2:
    rec = copy.deepcopy(rec)
    output = BatchRecurrentAttention16Layer_v2(
        input_size = rec._input_size,
        key_size = rec._key_size,
        value_size = rec._value_size,
        num_heads = rec._num_heads,
        ff_size = rec._ff_size,
        num_modules = rec._num_modules,
    )

    # Convert linear layers
    def convert(m_query, m_key, m_value, m_state):
        m = m_query
        if not isinstance(m, NonBatchLinear):
            return m
        return BatchLinear(
                m_query.to_linear_modules() + m_key.to_linear_modules() + m_value.to_linear_modules() + m_state.to_linear_modules(),
                default_batch=m.default_batch
        )
    output.fc_outputs = torch.nn.Sequential(*[
        convert(*x) for x in zip(rec.fc_query, rec.fc_key, rec.fc_value, rec.fc_state)
    ])
    output.fc_gates = torch.nn.Sequential(*[
        convert(*x) for x in zip(rec.fc_query_gate, rec.fc_key_gate, rec.fc_value_gate, rec.fc_state_gate)
    ])

    # Convert attention
    output.attention = BatchMultiHeadAttentionEinsum(
        rec.attention.to_multihead_attention_modules(),
        key_size=rec._key_size,
        num_heads=rec._num_heads,
        default_batch=rec.attention.default_batch,
    )

    # Copy default state
    output.default_state = rec.default_state

    return output


if __name__ == '__main__':
    # Performance tests
    import timeit

    if torch.cuda.is_available():
        print('CUDA available')
        device = torch.device('cuda')
        input_size = 512
        internal_size = 512
        num_heads = 8
        ff_size = []
        num_modules = 8
        num_iterations = 1_000
    else:
        print('CUDA not available')
        device = torch.device('cpu')
        input_size = 16
        internal_size = 16
        num_heads = 2
        ff_size = []
        num_modules = 2
        num_iterations = 5

    nonbatched_module = NonBatchRecurrentAttention16Layer(
        input_size=input_size,
        key_size=internal_size,
        value_size=internal_size,
        num_heads=num_heads,
        ff_size=ff_size,
        num_modules=num_modules,
    )
    batched_module = BatchRecurrentAttention16Layer.from_nonbatched(nonbatched_module)
    batched_module_v2 = BatchRecurrentAttention16Layer_v2.from_nonbatched(nonbatched_module)

    batched_module.to(device)
    batched_module_v2.to(device)
    nonbatched_module.to(device)

    batch_size = 8
    num_inputs = 8
    key = torch.randn([num_inputs, batch_size, input_size], device=device)
    value = torch.randn([num_inputs, batch_size, input_size], device=device)
    state = batched_module.init_state(batch_size)

    def test_batched():
        batched_module(state, key, value)
    def test_batched_v2():
        batched_module_v2(state, key, value)
    def test_nonbatched():
        nonbatched_module(state, key, value)

    print('-'*80)
    print(f'{num_iterations} iterations of batched, forward only')
    total_time = timeit.Timer(test_batched).timeit(number=num_iterations)
    print(f'{num_iterations} iterations took {total_time} seconds')
    print(f'{num_iterations / total_time} iterations per second')
    print(f'{total_time / num_iterations} seconds per iteration')

    print('-'*80)
    print(f'{num_iterations} iterations of batched v2, forward only')
    total_time = timeit.Timer(test_batched_v2).timeit(number=num_iterations)
    print(f'{num_iterations} iterations took {total_time} seconds')
    print(f'{num_iterations / total_time} iterations per second')
    print(f'{total_time / num_iterations} seconds per iteration')

    print('-'*80)
    print(f'{num_iterations} iterations of nonbatched, forward only')
    total_time = timeit.Timer(test_nonbatched).timeit(number=num_iterations)
    print(f'{num_iterations} iterations took {total_time} seconds')
    print(f'{num_iterations / total_time} iterations per second')
    print(f'{total_time / num_iterations} seconds per iteration')

    """
    Results on CPU:
        input_size = 16
        internal_size = 16
        num_heads = 2
        ff_size = []
        num_modules = 2
        num_iterations = 5

    --------------------------------------------------------------------------------
    5 iterations of batched, forward only
    5 iterations took 0.2044486680533737 seconds
    24.45601650334397 iterations per second
    0.04088973361067474 seconds per iteration
    --------------------------------------------------------------------------------
    5 iterations of nonbatched, forward only
    5 iterations took 0.017447802936658263 seconds
    286.56903210976077 iterations per second
    0.0034895605873316526 seconds per iteration
    """

    """
    Results on a Quadro RTX 6000 using BatchRecurrentAttention16Layer_v2:
        input_size = 512
        internal_size = 512
        num_heads = 8
        ff_size = [2048]
        num_modules = 4
        num_iterations = 1_000

    --------------------------------------------------------------------------------
    1000 iterations of batched, forward only
    1000 iterations took 1.1917530980426818 seconds
    839.0999793853154 iterations per second
    0.0011917530980426819 seconds per iteration
    --------------------------------------------------------------------------------
    1000 iterations of nonbatched, forward only
    1000 iterations took 10.057862909976393 seconds
    99.42469975486543 iterations per second
    0.010057862909976394 seconds per iteration
    """

    """
    Results on a Quadro RTX 6000 using BatchRecurrentAttention16Layer:
        input_size = 512
        internal_size = 512
        num_heads = 8
        ff_size = [2048]
        num_modules = 4
        num_iterations = 1_000

    --------------------------------------------------------------------------------
    1000 iterations of batched, forward only
    1000 iterations took 1.907696221023798 seconds
    524.1924730884735 iterations per second
    0.001907696221023798 seconds per iteration
    --------------------------------------------------------------------------------
    1000 iterations of nonbatched, forward only
    1000 iterations took 10.110477544134483 seconds
    98.90729647880406 iterations per second
    0.010110477544134482 seconds per iteration
    """

    # BatchRecurrentAttention16Layer_v2 is faster than NonBatchRecurrentAttention16Layer by about 60% (1.19s vs 1.91s).
    # The difference decreases as ff_size gets larger
    # Number of inputs, batch size, and num_modules don't seem to affect the difference
