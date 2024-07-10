import numpy as np
import torch
import gymnasium


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=torch.float)
    elif isinstance(x, (int, float)):
        return torch.tensor(x, device=device, dtype=torch.float)
    elif isinstance(x, dict):
        return {k: to_tensor(v, device) for k,v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_tensor(v, device) for v in x]
    elif isinstance(x, (list, list)):
        return [to_tensor(v, device) for v in x]
    else:
        raise ValueError(f'Unknown type: {type(x)}')


def reset_hidden(terminal, hidden, initial_hidden, batch_dim):
    assert len(hidden) == len(initial_hidden)
    assert len(hidden) == len(batch_dim)
    output = tuple([
        torch.where(terminal.view(-1, *([1]*(len(h.shape)-d-1))), init_h, h)
        for init_h,h,d in zip(initial_hidden, hidden, batch_dim)
    ])
    #for h,ih,o,t,d in zip(hidden, initial_hidden, output, terminal, batch_dim):
    #    assert list(h.shape) == list(ih.shape)
    #    assert list(h.shape) == list(o.shape)
    return output


def action_dist_discrete(net_output, n=None):
    dist = torch.distributions.Categorical(logits=net_output['action'][:n])
    return dist, dist.log_prob


def action_dist_continuous(net_output, n=None):
    action_mean = net_output['action_mean'][:n]
    action_logstd = net_output['action_logstd'][:n].clamp(-10, 10)
    dist = torch.distributions.Normal(action_mean, action_logstd.exp())
    return dist, lambda x: dist.log_prob(x).sum(-1)


def get_action_dist_function(action_space: gymnasium.Space):
    if isinstance(action_space, gymnasium.spaces.Discrete):
        return action_dist_discrete
    elif isinstance(action_space, gymnasium.spaces.Box):
        return action_dist_continuous
    else:
        raise NotImplementedError(f'Unknown action space: {action_space}')

