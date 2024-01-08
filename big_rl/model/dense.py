""" Utilities for making fully connected neural networks. """

import torch


def make_ff_model(input_size: int, output_size: int, hidden_size: list[int], activation = torch.nn.ReLU):
    modules = []
    for in_size, out_size in zip([input_size] + hidden_size, hidden_size + [output_size]):
        modules.append(torch.nn.Linear(in_size, out_size))
        if activation is not None:
            modules.append(activation())
    return torch.nn.Sequential(*modules[:-1])
