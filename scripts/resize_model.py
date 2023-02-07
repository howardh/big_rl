import argparse
import copy
import os

import torch

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model
#from big_rl.mujoco.envs import make_env
#from big_rl.minigrid.arguments import init_parser_model
#from big_rl.mujoco.common import env_config_presets, init_model

from big_rl.utils import merge_space
from big_rl.model.model import LinearOutput


def resize_ModularPolicy5(model, architecture) -> torch.nn.Module:
    model = copy.deepcopy(model)

    recurrence_type = model.attention[0].__class__.__name__
    if recurrence_type not in ['RecurrentAttention14']:
        raise NotImplementedError(f'Recurrence type {recurrence_type} not supported')

    attention = model._init_core_modules(
            recurrence_type = recurrence_type,
            input_size = model._input_size,
            key_size = model._key_size,
            value_size = model._value_size,
            num_heads = model._num_heads,
            ff_size = model._ff_size,
            architecture = architecture,
    )

    for i, (new_size, old_size) in enumerate(zip(architecture, model._architecture)):
        # Replace linear modules
        for seq_name in ['fc_key', 'fc_value', 'fc_output', 'fc_gate']:
            seq = getattr(model.attention[i], seq_name)
            if old_size < new_size:
                for j in [1,3]:
                    old_linear_modules = seq[j].to_linear_modules()
                    cls = type(seq[j])
                    in_size = old_linear_modules[0].in_features
                    out_size = old_linear_modules[0].out_features
                    new_linear_modules = old_linear_modules + [
                            torch.nn.Linear(in_size, out_size)
                            for _ in range(new_size - old_size)
                    ]
                    seq[j] = cls(new_linear_modules)
            else:
                for j in [1,3]:
                    old_linear_modules = seq[j].to_linear_modules()
                    cls = type(seq[j])
                    new_linear_modules = old_linear_modules[:new_size]
                    seq[j] = cls(new_linear_modules)

        # Replace attention modules
        cls = type(model.attention[i].attention)

        old_attn_modules = model.attention[i].attention.to_multihead_attention_modules()
        new_attn_modules = attention[i].attention.to_multihead_attention_modules()

        assert old_size == len(old_attn_modules)
        if old_size < new_size:
            new_attn_modules = old_attn_modules + new_attn_modules[old_size:]
        else:
            new_attn_modules = old_attn_modules[:new_size]
        model.attention[i].attention = cls(new_attn_modules, key_size=model._key_size, num_heads=model._num_heads)

    # Initial hidden state
    old_initial_hidden_state = model.initial_hidden_state
    new_initial_hidden_state = []
    for old_h,new_size in zip(old_initial_hidden_state, architecture):
        old_size = old_h.shape[0]
        if old_size < new_size:
            new_h = torch.zeros([new_size, model._input_size])
            new_h[:old_size] = old_h
            new_initial_hidden_state.append(new_h)
        else:
            new_initial_hidden_state.append(old_h[:new_size])
    model.initial_hidden_state = torch.nn.ParameterList([torch.nn.Parameter(h.clone()) for h in new_initial_hidden_state])

    return model


def resize_ModularPolicy5LSTM(model, hidden_size) -> torch.nn.Module:
    model = copy.deepcopy(model)

    input_size = model.lstm.input_size
    old_hidden_size = model.lstm.hidden_size
    new_lstm = torch.nn.LSTMCell(input_size, hidden_size)
    new_initial_hidden_state = [
        torch.zeros([hidden_size]),
        torch.zeros([hidden_size]),
    ]
    new_state_dict = new_lstm.state_dict()
    if old_hidden_size < hidden_size:
        size = model.lstm.weight_ih.shape[0]
        assert size == model.lstm.weight_hh.shape[0]
        assert size == model.lstm.bias_ih.shape[0]
        assert size == model.lstm.bias_hh.shape[0]

        new_state_dict["weight_ih"][:size,:model.lstm.weight_ih.shape[1]] = model.lstm.weight_ih
        new_state_dict["weight_hh"][:size,:model.lstm.weight_hh.shape[1]] = model.lstm.weight_hh
        new_state_dict["bias_ih"][:size] = model.lstm.bias_ih
        new_state_dict["bias_hh"][:size] = model.lstm.bias_hh
        new_initial_hidden_state[0][:old_hidden_size] = model.initial_hidden_state[0]
        new_initial_hidden_state[1][:old_hidden_size] = model.initial_hidden_state[1]
    else:
        size = new_lstm.weight_ih.shape[0]
        assert size == new_lstm.weight_hh.shape[0]
        assert size == new_lstm.bias_ih.shape[0]
        assert size == new_lstm.bias_hh.shape[0]

        new_state_dict["weight_ih"] = model.lstm.weight_ih[:size,:new_lstm.weight_ih.shape[1]]
        new_state_dict["weight_hh"] = model.lstm.weight_hh[:size,:new_lstm.weight_hh.shape[1]]
        new_state_dict["bias_ih"] = model.lstm.bias_ih[:size]
        new_state_dict["bias_hh"] = model.lstm.bias_hh[:size]
        new_initial_hidden_state[0] = model.initial_hidden_state[0][:hidden_size]
        new_initial_hidden_state[1] = model.initial_hidden_state[1][:hidden_size]
    new_lstm.load_state_dict(new_state_dict)

    model.lstm = new_lstm
    model.initial_hidden_state = torch.nn.ParameterList([
        torch.nn.Parameter(p) for p in new_initial_hidden_state])

    # Resize output modules
    for k, v in model.output_modules.items():
        if isinstance(v, LinearOutput):
            model.output_modules[k] = resize_LinearOutput(v, input_size=hidden_size)
        else:
            raise NotImplementedError(f'Output module {v} not supported')

    return model


def resize_LinearOutput(output_module, input_size) -> torch.nn.Module:
    output_size = output_module.output_size
    old_input_size = output_module.query.size(0)
    num_heads = output_module.attention.num_heads
    new_output_module = LinearOutput(
        output_size=output_size,
        key_size=input_size,
        num_heads=num_heads,
    )
    new_state_dict = new_output_module.state_dict()
    if old_input_size < input_size:
        new_state_dict["query"][:old_input_size] = output_module.query
        size = output_module.attention.in_proj_bias.shape[0]
        new_state_dict["attention.in_proj_weight"][:size,:old_input_size] = output_module.attention.in_proj_weight
        new_state_dict["attention.in_proj_bias"][:size] = output_module.attention.in_proj_bias
        new_state_dict["attention.out_proj.weight"][:old_input_size,:old_input_size] = output_module.attention.out_proj.weight
        new_state_dict["attention.out_proj.bias"][:old_input_size] = output_module.attention.out_proj.bias
        new_state_dict["ff.weight"][:,:old_input_size] = output_module.ff.weight
        new_state_dict["ff.bias"][:] = output_module.ff.bias
    else:
        raise NotImplementedError('Not implemented yet')
    new_output_module.load_state_dict(new_state_dict)
    #for (old_n,old_p), (new_n,new_p) in zip(output_module.named_parameters(), new_output_module.named_parameters()):
    #    assert old_n == new_n
    #    print(old_n, old_p.shape, new_p.shape)
    #assert False
    #breakpoint()
    return new_output_module


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=str, nargs='+',
                        help='Environments whose observation and action spaces are used to initialize the model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model checkpoint to resize.')
    parser.add_argument('--output-model', type=str, default=None,
                        help='Path to save the resized model.')
    parser.add_argument('--target-architecture', type=int, nargs='+',
                        help='Architecture of the model after resizing.')
    parser.add_argument('--target-hidden-size', type=int,
                        help='Hidden size of the model after resizing. Only applies to LSTM.')
    init_parser_model(parser)
    args = parser.parse_args()

    # Validate arguments
    if args.model is None:
        raise ValueError('Please specify a model to resize with --model.')
    if args.output_model is None:
        raise ValueError('Please specify a path to save the resized model with --output-model.')
    if os.path.exists(args.output_model):
        raise ValueError(f'Output model path {args.output_model} already exists.')

    # Create environment
    ENV_CONFIG_PRESETS = env_config_presets()
    env_configs = [ ENV_CONFIG_PRESETS[e] for e in args.envs ]
    envs = [ make_env(**conf) for conf in env_configs ]

    # Initialize model
    model = init_model(
            observation_space = merge_space(*[env.observation_space for env in envs]),
            action_space = envs[0].action_space, # Assume the same action space for all environments
            model_type = args.model_type,
            recurrence_type = args.recurrence_type,
            architecture = args.architecture,
            hidden_size = args.hidden_size,
            device = torch.device('cpu'),
    )

    checkpoint = torch.load(args.model, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded checkpoint from {args.model}')

    # Resize model
    # TODO: Resizing for other models

    checkpoint['model'] = resize_ModularPolicy5LSTM(model, args.target_hidden_size).state_dict()
    del checkpoint['optimizer']
    torch.save(checkpoint, args.output_model)
    print(f'Saved checkpoint to {args.output_model}')
