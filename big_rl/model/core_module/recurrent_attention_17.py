"""
Changes from #16 -> #17:
- Modified to fit with the CoreModule API
  - Renamed `state` to `hidden` and changed it to be the last argument
  - Modified to only allow one layer, since the container can stack them
- Modified gating to handle each element separately instead of gating the entire vector with one output. This allows for more efficient paralellization.
- Modified the final layer of the fully connected layers so they don't go through two activation functions in a row.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Tuple, List, Type

import torch
from torchtyping.tensor_type import TensorType
from big_rl.model.core_module.container import CoreModule

from big_rl.model.model import BatchMultiHeadAttentionEinsum, NonBatchMultiHeadAttention, BatchMultiHeadAttentionBroadcast
from big_rl.model.model import BatchLinear, NonBatchLinear


class RecurrentAttention17(CoreModule):
    def __init__(self, key_size: int, value_size: int, num_heads: int, ff_size: list[int] = [], num_modules: int = 1, batch_type: str = 'einsum'):
        torch.nn.Module.__init__(self)

        # Preprocess and validate parameters
        if key_size != value_size:
            raise ValueError('Key size must equal value size')
        if isinstance(ff_size, int):
            ff_size = [ff_size]

        self._input_size = key_size

        # Save parameters
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._ff_size = ff_size
        self._num_modules = num_modules

        # Initialize fully-connected modules
        self.fc = self._make_mlp(
                [self._input_size*2, *ff_size, value_size],
                num_duplicates=4*2,
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

        # Initialize default hidden
        self.default_hidden = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros([num_modules, self._input_size])),
                torch.nn.Parameter(torch.zeros([num_modules, key_size])),
                torch.nn.Parameter(torch.zeros([num_modules, key_size])),
                torch.nn.Parameter(torch.zeros([num_modules, value_size])),
        ])

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
            hidden: Tuple[
                TensorType['num_blocks','batch_size','input_size',float], # Internal state
                TensorType['num_blocks','batch_size','key_size',float], # Previous query
                TensorType['num_blocks','batch_size','key_size',float], # Previous key
                TensorType['num_blocks','batch_size','value_size',float], # Previous value
            ],
        ):
        num_modules = self._num_modules
        batch_size = key.shape[1]
        input_size = hidden[0].size(2)
        assert num_modules == hidden[0].size(0)
        assert len(hidden) == 4

        prev_hidden = hidden[0] # (num_blocks, batch_size, input_size)
        prev_query = hidden[1] # (num_blocks, batch_size, input_size)
        prev_key = hidden[2] # (num_blocks, batch_size, key_size)
        prev_value = hidden[3] # (num_blocks, batch_size, value_size)

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

        fc_input = torch.cat([attn_output, prev_hidden], dim=2)
        fc_input = fc_input.unsqueeze(0).expand(8, num_modules, batch_size, input_size*2).reshape(num_modules*8, batch_size, input_size*2)

        fc_output = self.fc(fc_input) # (num_blocks*4*2, batch_size, key_size)
        split_idx = fc_output.shape[0] // 2 # num_blocks*4. This is the index that separates the output values and the gate values

        outputs = fc_output[:split_idx].tanh() # (num_blocks*4, batch_size, key_size)
        #foo = outputs[:2]
        outputs = outputs.view([4, num_modules, batch_size, -1]) # (num_blocks, 4, batch_size, size)
        output_queries = outputs[0] # (num_blocks, batch_size, key_size)
        output_keys = outputs[1] # (num_blocks, batch_size, key_size)
        output_values = outputs[2] # (num_blocks, batch_size, value_size)
        output_hidden = outputs[3] # (num_blocks, batch_size, input_size)
        #breakpoint()

        gates = fc_output[split_idx:].sigmoid() # (num_blocks*4, batch_size)
        output_query_gate, output_key_gate, output_value_gate, output_hidden_gate = gates.view(4, num_modules, batch_size, -1)

        gated_output_queries = output_query_gate * output_queries + (1 - output_query_gate) * prev_query
        gated_output_keys = output_key_gate * output_keys + (1 - output_key_gate) * prev_key
        gated_output_values = output_value_gate * output_values + (1 - output_value_gate) * prev_value
        gated_output_hidden = output_hidden_gate * output_hidden + (1 - output_hidden_gate) * prev_hidden

        return { # seq_len = number of inputs receives
            'key': gated_output_keys, # (num_blocks, batch_size, key_size)
            'value': gated_output_values, # (num_blocks, batch_size, value_size)
            'hidden': (
                gated_output_hidden, # (num_blocks, batch_size, input_size)
                gated_output_queries, # (num_blocks, batch_size, key_size)
                gated_output_keys, # (num_blocks, batch_size, key_size)
                gated_output_values, # (num_blocks, batch_size, value_size)
            ),
            'misc': {
                'attn_output': attn_output, # (num_blocks, batch_size, value_size)
                'attn_output_weights': attn_output_weights, # (num_blocks, batch_size, seq_len)
                'gates': {
                    'query': output_query_gate, # (num_blocks, batch_size)
                    'key': output_key_gate, # (num_blocks, batch_size)
                    'value': output_value_gate, # (num_blocks, batch_size)
                    'hidden': output_hidden_gate, # (num_blocks, batch_size)
                },
            },
        }

    def _make_mlp(self, sizes, num_duplicates=1):
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.ReLU())
            layers.append(
                BatchLinear([
                    torch.nn.Linear(in_size, out_size) for _ in range(self._num_modules*num_duplicates)
                ], default_batch=True),
            )
        return torch.nn.Sequential(*layers)

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_hidden
        )

    @property
    def n_hidden(self):
        return len(self.default_hidden)

    def to_nonbatched(self):
        return _batched_to_nonbatched(self)

    @classmethod
    def from_nonbatched(cls, obj) -> 'BatchRecurrentAttention17':
        return _to_batched(obj)
BatchRecurrentAttention17 = RecurrentAttention17


class NonBatchRecurrentAttention17(RecurrentAttention17):
    """ Same as RecurrentAttention14, but added gating to the output keys and values. 
    
    API changes:
    - `forward()` takes three inputs: hidden, key, value. It also outputs a dictionary with the same three keys.
        - Debugging values: `attn_output`, `attn_output_weights`, `gates`
    - `init_hidden(batch_size)` returns a tuple which is used to initialize the state.

    Previously, `forward()` had a `initial_x` input which was used as a default query. This doesn't change between batches, so it makes more sense for this to be a parameter of the model than an input. Unclear why I made it a parameter of the parent module rather than of the recurrence module.
    """
    def __init__(self, key_size, value_size, num_heads, ff_size: list[int] = [], num_modules: int = 1):
        torch.nn.Module.__init__(self)

        # Preprocess parameters
        if key_size != value_size:
            raise ValueError('Key size must equal value size')
        if isinstance(ff_size, int):
            ff_size = [ff_size]

        # Save parameters
        self._input_size = key_size
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._ff_size = ff_size
        self._num_modules = num_modules
        input_size = key_size

        # Initialize fully-connected modules
        self.fc_query = self._make_mlp([input_size*2, *ff_size, key_size])
        self.fc_key = self._make_mlp([input_size*2, *ff_size, key_size])
        self.fc_value = self._make_mlp([input_size*2, *ff_size, value_size])
        self.fc_hidden = self._make_mlp([input_size*2, *ff_size, input_size])

        self.fc_query_gate = self._make_mlp(
                [input_size*2, *ff_size, key_size],
        )
        self.fc_key_gate = self._make_mlp(
                [input_size*2, *ff_size, key_size],
        )
        self.fc_value_gate = self._make_mlp(
                [input_size*2, *ff_size, key_size],
        )
        self.fc_hidden_gate = self._make_mlp(
                [input_size*2, *ff_size, key_size],
        )

        # Initialize attention module
        self.attention = NonBatchMultiHeadAttention([
            torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
            for _ in range(num_modules)
        ], key_size=key_size, num_heads=num_heads, default_batch=True)

        # Initialize default hidden
        self.default_hidden = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros([num_modules, input_size])),
                torch.nn.Parameter(torch.zeros([num_modules, key_size])),
                torch.nn.Parameter(torch.zeros([num_modules, key_size])),
                torch.nn.Parameter(torch.zeros([num_modules, value_size])),
        ])

    def forward(self,
            key: TensorType['seq_len','batch_size','key_size',float],
            value: TensorType['seq_len','batch_size','value_size',float],
            hidden: Tuple[
                TensorType['num_blocks','batch_size','input_size',float], # Internal state
                TensorType['num_blocks','batch_size','key_size',float], # Previous query
                TensorType['num_blocks','batch_size','key_size',float], # Previous key
                TensorType['num_blocks','batch_size','value_size',float], # Previous value
            ],
        ):
        num_modules = self._num_modules
        assert num_modules == hidden[0].size(0)
        assert len(hidden) == 4

        prev_hidden = hidden[0] # (num_blocks, batch_size, input_size)
        prev_query = hidden[1] # (num_blocks, batch_size, input_size)
        prev_key = hidden[2] # (num_blocks, batch_size, key_size)
        prev_value = hidden[3] # (num_blocks, batch_size, value_size)

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

        fc_input = torch.cat([attn_output, prev_hidden], dim=2)

        output_queries = self.fc_query(fc_input).tanh() # (num_blocks, batch_size, key_size)
        output_keys = self.fc_key(fc_input).tanh() # (num_blocks, batch_size, key_size)
        output_values = self.fc_value(fc_input).tanh() # (num_blocks, batch_size, value_size)
        output_hidden = self.fc_hidden(fc_input).tanh() # (num_blocks, batch_size, input_size)

        output_query_gate = self.fc_query_gate(fc_input).sigmoid() # (num_blocks, batch_size)
        output_key_gate = self.fc_key_gate(fc_input).sigmoid() # (num_blocks, batch_size)
        output_value_gate = self.fc_value_gate(fc_input).sigmoid() # (num_blocks, batch_size)
        output_hidden_gate = self.fc_hidden_gate(fc_input).sigmoid() # (num_blocks, batch_size)

        gated_output_queries = output_query_gate * output_queries + (1 - output_query_gate) * prev_query
        gated_output_keys = output_key_gate * output_keys + (1 - output_key_gate) * prev_key
        gated_output_values = output_value_gate * output_values + (1 - output_value_gate) * prev_value
        gated_output_hidden = output_hidden_gate * output_hidden + (1 - output_hidden_gate) * prev_hidden

        return { # seq_len = number of inputs receives
            'attn_output': attn_output, # (num_blocks, batch_size, value_size)
            'attn_output_weights': attn_output_weights, # (num_blocks, batch_size, seq_len)
            'gates': {
                'query': output_query_gate, # (num_blocks, batch_size)
                'key': output_key_gate, # (num_blocks, batch_size)
                'value': output_value_gate, # (num_blocks, batch_size)
                'hidden': output_hidden_gate, # (num_blocks, batch_size)
            },
            'key': gated_output_keys, # (num_blocks, batch_size, key_size)
            'value': gated_output_values, # (num_blocks, batch_size, value_size)
            'hidden': (
                gated_output_hidden, # (num_blocks, batch_size, input_size)
                gated_output_queries, # (num_blocks, batch_size, key_size)
                gated_output_keys, # (num_blocks, batch_size, key_size)
                gated_output_values, # (num_blocks, batch_size, value_size)
            )
        }

    def _make_mlp(self, sizes):
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.ReLU())
            layers.append(
                NonBatchLinear([
                    torch.nn.Linear(in_size, out_size) for _ in range(self._num_modules)
                ], default_batch=True),
            )
        return torch.nn.Sequential(*layers)

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor, ...]:
        return tuple(
            x.unsqueeze(1).expand([self._num_modules, batch_size, x.shape[1]])
            for x in self.default_hidden
        )

    @property
    def n_hidden(self):
        return len(self.default_hidden)

    def to_nonbatched(self):
        return self

    @classmethod
    def from_nonbatched(cls, obj) -> 'NonBatchRecurrentAttention17':
        return obj

    def remove_modules(self, mask):
        assert len(mask) == self._num_modules, f"Mask must be the same length as the number of modules. Expected {self._num_modules}. Received {len(mask)}"

        def process(module):
            if isinstance(module, NonBatchLinear):
                return NonBatchLinear(
                    [l for l, m in zip(module.to_linear_modules(), mask) if m],
                    default_batch=module.default_batch,
                )
            else:
                return module

        self.fc_query = torch.nn.Sequential(*[process(m) for m in self.fc_query])
        self.fc_key = torch.nn.Sequential(*[process(m) for m in self.fc_key])
        self.fc_value = torch.nn.Sequential(*[process(m) for m in self.fc_value])
        self.fc_hidden = torch.nn.Sequential(*[process(m) for m in self.fc_hidden])

        self.fc_query_gate = torch.nn.Sequential(*[process(m) for m in self.fc_query_gate])
        self.fc_key_gate = torch.nn.Sequential(*[process(m) for m in self.fc_key_gate])
        self.fc_value_gate = torch.nn.Sequential(*[process(m) for m in self.fc_value_gate])
        self.fc_hidden_gate = torch.nn.Sequential(*[process(m) for m in self.fc_hidden_gate])

        self.attention = NonBatchMultiHeadAttention([
            a for a,m in zip(self.attention.to_multihead_attention_modules(), mask) if m
        ], key_size=self._key_size, num_heads=self._num_heads, default_batch=self.attention.default_batch)

        self.default_hidden = torch.nn.ParameterList([
            torch.nn.Parameter(s[torch.tensor(mask),:])
            for s in self.default_hidden
        ])

        self._num_modules = sum(mask)

    def merge(self, module: NonBatchRecurrentAttention17, positions=None):
        def interleave(a:list,b:list):
            if positions is None:
                return a+b
            else:
                output = []
                for i in range(len(a)+len(b)):
                    if i in positions:
                        output.append(b.pop(0))
                    else:
                        output.append(a.pop(0))
                return output
        def process(module1,module2):
            assert type(module1) == type(module2)

            if isinstance(module1, NonBatchLinear) and isinstance(module2, NonBatchLinear):
                assert module1.default_batch == module2.default_batch
                return NonBatchLinear(
                    interleave(module1.to_linear_modules(), module2.to_linear_modules()),
                    default_batch=module1.default_batch,
                )
            else:
                return module1

        self.fc_query = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_query, module.fc_query)])
        self.fc_key = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_key, module.fc_key)])
        self.fc_value = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_value, module.fc_value)])
        self.fc_hidden = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_hidden, module.fc_hidden)])

        self.fc_query_gate = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_query_gate, module.fc_query_gate)])
        self.fc_key_gate = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_key_gate, module.fc_key_gate)])
        self.fc_value_gate = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_value_gate, module.fc_value_gate)])
        self.fc_hidden_gate = torch.nn.Sequential(*[process(m1,m2) for m1,m2 in zip(self.fc_hidden_gate, module.fc_hidden_gate)])

        self.attention = NonBatchMultiHeadAttention(
            interleave(
                self.attention.to_multihead_attention_modules(),
                module.attention.to_multihead_attention_modules()
            ),
            key_size=self._key_size,
            num_heads=self._num_heads,
            default_batch=self.attention.default_batch
        )

        self.default_hidden = torch.nn.ParameterList([
            torch.nn.Parameter(torch.cat(interleave(list(s1.split(1)),list(s2.split(1))),dim=0))
            for s1,s2 in zip(self.default_hidden, module.default_hidden)
        ])

        self._num_modules += module._num_modules


# Batching / Unbatching utils


def _batched_to_nonbatched(rec: BatchRecurrentAttention17) -> NonBatchRecurrentAttention17:
    rec = copy.deepcopy(rec)
    output = NonBatchRecurrentAttention17(
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
        hidden = []
        for m in sequential:
            if not isinstance(m, BatchLinear):
                query.append(m)
                key.append(m)
                value.append(m)
                hidden.append(m)
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
            hidden.append(NonBatchLinear(
                    linear_modules[3*n:],
                    default_batch=m.default_batch
            ))
        return torch.nn.Sequential(*query), torch.nn.Sequential(*key), torch.nn.Sequential(*value), torch.nn.Sequential(*hidden)

    output.fc_query, output.fc_key, output.fc_value, output.fc_hidden = convert(rec.fc_outputs)
    output.fc_query_gate, output.fc_key_gate, output.fc_value_gate, output.fc_hidden_gate = convert(rec.fc_gates)

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

    # Copy default hidden
    output.default_hidden = rec.default_hidden

    return output


def _to_batched(rec: NonBatchRecurrentAttention17) -> BatchRecurrentAttention17:
    rec = copy.deepcopy(rec)
    output = BatchRecurrentAttention17(
        key_size = rec._key_size,
        value_size = rec._value_size,
        num_heads = rec._num_heads,
        ff_size = rec._ff_size,
        num_modules = rec._num_modules,
    )

    # Convert linear layers
    def convert(m_query, m_key, m_value, m_hidden):
        m = m_query
        if not isinstance(m, NonBatchLinear):
            return m
        return BatchLinear(
                m_query.to_linear_modules() + m_key.to_linear_modules() + m_value.to_linear_modules() + m_hidden.to_linear_modules(),
                default_batch=m.default_batch
        )
    output.fc_outputs = torch.nn.Sequential(*[
        convert(*x) for x in zip(rec.fc_query, rec.fc_key, rec.fc_value, rec.fc_hidden)
    ])
    output.fc_gates = torch.nn.Sequential(*[
        convert(*x) for x in zip(rec.fc_query_gate, rec.fc_key_gate, rec.fc_value_gate, rec.fc_hidden_gate)
    ])

    # Convert attention
    output.attention = BatchMultiHeadAttentionEinsum(
        rec.attention.to_multihead_attention_modules(),
        key_size=rec._key_size,
        num_heads=rec._num_heads,
        default_batch=rec.attention.default_batch,
    )

    # Copy default hidden
    output.default_hidden = rec.default_hidden

    return output


# Resizing / Ablation / Merging utils


def ablate(model: RecurrentAttention17, mask: List[bool]) -> RecurrentAttention17:
    """ Return a new RecurrentAttention17 model with the specified modules removed.
    """

    # Validation
    if len(mask) != model._num_modules:
        raise ValueError(f"Mask length ({len(mask)}) must match number of modules ({model._num_modules})")

    # Convert to non-batched layers
    model = copy.deepcopy(model)
    original_model_type = type(model._layers)
    model = model.to_nonbatched()

    assert isinstance(model, NonBatchRecurrentAttention17)
    model.remove_modules(mask)

    # Convert back to original type
    assert isinstance(original_model_type, RecurrentAttention17)
    model = original_model_type.from_nonbatched(model)

    return model


def merge(models: List[RecurrentAttention17]):
    ...


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

    nonbatched_module = NonBatchRecurrentAttention17(
        key_size=internal_size,
        value_size=internal_size,
        num_heads=num_heads,
        ff_size=ff_size,
        num_modules=num_modules,
    )
    batched_module = BatchRecurrentAttention17.from_nonbatched(nonbatched_module)
    batched_module_v2 = BatchRecurrentAttention17.from_nonbatched(nonbatched_module)

    batched_module.to(device)
    batched_module_v2.to(device)
    nonbatched_module.to(device)

    batch_size = 8
    num_inputs = 8
    key = torch.randn([num_inputs, batch_size, input_size], device=device)
    value = torch.randn([num_inputs, batch_size, input_size], device=device)
    hidden = batched_module.init_hidden(batch_size)

    def test_batched():
        batched_module(key, value, hidden)
    def test_batched_v2():
        batched_module_v2(key, value, hidden)
    def test_nonbatched():
        nonbatched_module(key, value, hidden)

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
