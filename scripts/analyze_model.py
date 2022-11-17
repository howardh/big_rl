import argparse

import torch

from big_rl.minigrid.envs import make_env
from big_rl.minigrid.arguments import init_parser_model
from big_rl.minigrid.common import env_config_presets, init_model

def count_parameters(parameters):
    return sum(p.numel() for p in parameters if p.requires_grad)


def equivalent_LSTM(num_parameters, input_size):
    # Size of an LSTM is 4hs+4hh+4h*2 where s is the input size, h is the hidden size (According to the "Variables" section of https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell)
    # p = 4hs+4hh+4h*2
    #   = 4hs + 4hh + 8h
    # 0 = 4h^2 + (4s+8)h - p
    # h = (-4s-8 +- sqrt((4s+8)^2 - 16*4p)) / 8

    a = 4
    b = 4 * input_size + 8
    c = -num_parameters
    hidden_size = int((-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a))
    lstm = torch.nn.LSTMCell(
            input_size = input_size,
            hidden_size = hidden_size,
    )

    return lstm


def equivalent_GRU(num_parameters, input_size):
    # Size of a GRU is 3hs+3hh+3h*2 where s is the input size, h is the hidden size (According to the "Variables" section of https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell)
    # p = 3hs+3hh+3h*2
    #   = 3hs + 3hh + 6h
    # 0 = 3h^2 + (3s+6)h - p

    a = 3
    b = 3 * input_size + 6
    c = -num_parameters
    hidden_size = int((-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a))
    gru = torch.nn.GRUCell(
            input_size = input_size,
            hidden_size = hidden_size,
    )

    return gru


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        help='Environment whose observation and action spaces are used to initialize the model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a model checkpoint to analyze.')
    init_parser_model(parser)
    args = parser.parse_args()

    # Create environment
    ENV_CONFIG_PRESETS = env_config_presets()
    env_config = ENV_CONFIG_PRESETS[args.env]

    env = make_env(**env_config)

    # Initialize model
    model = init_model(
            observation_space = env.observation_space,
            action_space = env.action_space,
            model_type = args.model_type,
            recurrence_type = args.recurrence_type,
            architecture = args.architecture,
            device = torch.device('cpu'),
    )

    # Count number of parameters
    print()
    print(f'Model type: {args.model_type}')
    print(f'Recurrence type: {args.recurrence_type}')
    print(f'Architecture: {args.architecture}')

    print('-' * 80)

    print('Number of parameters')
    print(f'\tTotal: {count_parameters(model.parameters()):,}')
    print(f'\tInput modules')
    for k,v in model.input_modules.items():
        print(f'\t\t{k}: {count_parameters(v.parameters()):,}')
    print(f'\tCore modules: {count_parameters(model.attention.parameters()):,}')
    print(f'\tOutput modules')
    for k,v in model.output_modules.items():
        print(f'\t\t{k}: {count_parameters(v.parameters()):,}')
    print(f'\tHidden size: {sum(p.numel() for p in model.init_hidden(1)):,}')

    print('-' * 80)

    print('LSTM')
    lstm = equivalent_LSTM(
            num_parameters = count_parameters(model.attention.parameters()),
            input_size = 512 * len(model.input_modules)
    )
    print(f'\tSize of an equivalent LSTM: {count_parameters(lstm.parameters()):,}')
    print(f'\tInput size: {lstm.input_size}')
    print(f'\tHidden size: {lstm.hidden_size}')

    print('GRU')
    gru = equivalent_GRU(
            num_parameters = count_parameters(model.attention.parameters()),
            input_size = 512 * len(model.input_modules)
    )
    print(f'\tSize of an equivalent GRU: {count_parameters(gru.parameters()):,}')
    print(f'\tInput size: {lstm.input_size}')
    print(f'\tHidden size: {gru.hidden_size}')

    print()

    #breakpoint()
