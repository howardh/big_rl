import argparse
import copy
import itertools
import os
import time
import textwrap
import random

import gymnasium as gym
from gymnasium.vector import VectorEnv, AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, NormalizeObservation, TransformObservation, NormalizeReward, TransformReward, TimeLimit # pyright: ignore[reportPrivateImportUsage]
import torch
from torch.func import functional_call # pyright: ignore[reportPrivateImportUsage]
import numpy as np
import wandb
from tqdm import tqdm

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
import big_rl.mujoco.envs


doot = [0,0]


class Model(torch.nn.Module):
    """
    MAML paper model specifications:
    - Hidden layer size: 100
    - Number of hidden layers: 2
    - Activation function: relu
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.action_size = action_size
        self.v = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=1),
        )
        self.pi = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=action_size*2),
        )
    def forward(self, x):
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi,
                'action_mean': pi[..., :self.action_size],
                'action_logstd': pi[..., self.action_size:],
        }


class Model2(torch.nn.Module):
    """
    State independent logstd
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.action_size = action_size
        self.v = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=1),
        )
        self.pi_mean = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=action_size),
        )
        self.pi_logstd = torch.nn.Parameter(torch.zeros(action_size))
    def forward(self, x):
        batch_size = x.shape[:-1]
        unsqueeze_dims = [None] * len(batch_size)
        return {
                'value': self.v(x),
                #'action': pi,
                'action_mean': self.pi_mean(x),
                #'action_logstd': self.pi_logstd[*unsqueeze_dims, :].expand(*batch_size, -1).clamp(-10,None), # XXX: Requires python>=3.11
                'action_logstd': self.pi_logstd.__getitem__([*unsqueeze_dims, slice(None)]).expand(*batch_size, -1).clamp(-10,None),
        }


class Model3(torch.nn.Module):
    """
    State independent logstd
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.action_size = action_size
        self.pi_mean = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100,out_features=action_size),
        )
        self.pi_logstd = torch.nn.Parameter(torch.zeros(action_size))
    def forward(self, x):
        batch_size = x.shape[:-1]
        unsqueeze_dims = [None] * len(batch_size)
        output = {
                'action_mean': self.pi_mean(x),
                #'action_logstd': self.pi_logstd[*unsqueeze_dims, :].expand(*batch_size, -1).clamp(-10,None), # XXX: Requires python>=3.11
                'action_logstd': self.pi_logstd.__getitem__([*unsqueeze_dims, slice(None)]).expand(*batch_size, -1).clamp(-10,None),
        }
        return output


class Model4(torch.nn.Module):
    """
    State independent logstd
    """
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.action_size = action_size
        self.pi_mean = torch.nn.Sequential(
            torch.nn.Linear(in_features=obs_size,out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048,out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048,out_features=2048),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2048,out_features=action_size),
        )
        self.pi_logstd = torch.nn.Parameter(torch.zeros(action_size))
    def forward(self, x):
        batch_size = x.shape[:-1]
        unsqueeze_dims = [None] * len(batch_size)
        output = {
                'action_mean': self.pi_mean(x),
                #'action_logstd': self.pi_logstd[*unsqueeze_dims, :].expand(*batch_size, -1).clamp(-10,None), # XXX: Requires python>=3.11
                'action_logstd': self.pi_logstd.__getitem__([*unsqueeze_dims, slice(None)]).expand(*batch_size, -1).clamp(-10,None),
        }
        return output


def compute_advantage(obs, timesteps, reward, terminal, discount, gae_lambda):
    def vf_features(x, t):
        batch_size = x.shape[:-1]
        device = x.device
        t = t.unsqueeze(-1) / 100
        return torch.cat([x, x**2, t, t**2, t**3, torch.ones([*batch_size, 1], device=device)], dim=-1).float()
    def fit_vf(x, t, y):
        y = y.flatten()
        features = vf_features(x, t)
        features = features.flatten(end_dim=-2)
        vf_weights = torch.linalg.lstsq(features, y).solution
        return vf_weights

    returns = generalized_advantage_estimate(
            state_values = torch.zeros_like(reward[1:,:]),
            next_state_values = torch.zeros_like(reward[1:,:]),
            rewards = reward[1:,:],
            terminals = terminal[1:,:],
            discount = discount,
            gae_lambda = 1.0,
    )
    vf_weights = fit_vf(obs[:-1], timesteps[:-1], returns)
    vf = vf_features(obs, timesteps) @ vf_weights

    advantages = generalized_advantage_estimate(
            state_values = vf[:-1,:],
            next_state_values = vf[1:,:],
            rewards = reward[1:,:],
            terminals = terminal[1:,:],
            discount = discount,
            gae_lambda = gae_lambda,
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def update_trpo(
        data : list[tuple[VecHistoryBuffer,VecHistoryBuffer]],
        model : torch.nn.Module,
        initial_params : dict[str,torch.Tensor],
        adapted_params : list[dict[str,torch.Tensor]],
        inner_lr : float,
        discount : float,
        gae_lambda : float,
        target_kl : float):
    start_time = time.time()
    cg_damping = 1e-2
    device = next(model.parameters()).device

    key_order = [k for k,_ in model.named_parameters()]
    output = {}

    def surrogate_objective_dict(adapted_params):
        losses = []
        for (_, val_data), params in zip(data, adapted_params):
            n = len(val_data.obs_history)
            obs = torch.tensor(val_data.obs, dtype=torch.float, device=device)
            action = val_data.action
            reward = val_data.reward
            terminal = val_data.terminal
            old_log_action_probs = val_data.misc['log_action_prob'][1:] # type: ignore
            timesteps = val_data.misc['time'] # type: ignore

            net_output = functional_call(model, params, obs)
            action_mean = net_output['action_mean'][:n-1]
            action_logstd = net_output['action_logstd'][:n-1]
            action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
            log_action_probs = action_dist.log_prob(action).sum(-1)

            with torch.no_grad():
                advantage = compute_advantage(
                    obs = obs,
                    timesteps = timesteps,
                    reward = reward,
                    terminal = terminal,
                    discount = discount,
                    gae_lambda = gae_lambda,
                )

            loss = -adaptation_loss(
                log_action_probs=log_action_probs,
                old_log_action_probs=old_log_action_probs,
                returns=advantage.detach(),
                terminals=terminal[:n-1,:],
            ).mean()

            losses.append(loss)

        return torch.stack(losses).mean()
    
    def kl_divergence(model_params_dict):
        kl_divs = []
        for history, _ in data:
            n = len(history.obs_history)
            obs = torch.tensor(history.obs, dtype=torch.float, device=device)

            action_logstd_old = torch.tensor(history.misc['action_logstd'][1:], dtype=torch.float, device=device) # type: ignore
            action_mean_old = torch.tensor(history.misc['action_mean'][1:], dtype=torch.float, device=device) # type: ignore
            action_dist_old = torch.distributions.Normal(action_mean_old, action_logstd_old.exp())

            net_output = functional_call(model, model_params_dict, obs)
            action_mean = net_output['action_mean'][:n-1]
            action_logstd = net_output['action_logstd'][:n-1]
            # See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
            # For the KL divergence between two normal distributions
            #kl = action_logstd - action_logstd_old + (action_logstd_old.exp()**2 + (action_mean_old - action_mean)**2)/(2*action_logstd.exp()**2) - 0.5
            action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
            kl = torch.distributions.kl.kl_divergence(action_dist_old, action_dist)

            kl_divs.append(kl.sum(-1).mean())
        return torch.stack(kl_divs).mean()

    def conjugate_gradient_jvp(f, b, x=None, max_iter=10, tol=1e-3, argnums=0):
        """Solution of the linear system (Jf(z))x = b using the conjugate gradient method.

        Args:
            f (callable): A function that takes a tensor and returns a Jacobian-vector product with its input as the tangent.
            b (torch.Tensor): A vector.
            x (torch.Tensor): An initial guess for the solution. If None, then defaults to the zero vector.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for the residual.

        Returns:
            torch.Tensor: The solution (x) of the linear system f(x) = b.
        """
        if x is None:
            x = torch.zeros_like(b)
        #r = b - A @ x
        r = b - f(x)
        p = r.clone()
        for _ in range(max_iter):
            #Ap = A @ p
            Ap = f(p)
            alpha = (r @ r) / (p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            if torch.norm(r) < tol:
                break
            beta = (r @ r) / (p @ p)
            p = r + beta * p
        return x

    obj = surrogate_objective_dict(adapted_params)
    old_loss = -obj.item()
    obj_grad = torch.autograd.grad(obj, initial_params.values(), create_graph=True, allow_unused=True) # type: ignore
    obj_grad_flat = torch.cat([g.flatten() for g in obj_grad if g is not None]).detach()
    used_params = [g is not None for g in obj_grad]
    kl = kl_divergence(initial_params)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, initial_params.values(), create_graph=True, allow_unused=True) # type: ignore
    kl_grad_flat = torch.cat([g.flatten() for g in kl_grad if g is not None])
    def fisher_vector_product(vec):
        kl_grad_vec = kl_grad_flat @ vec
        fvp = torch.autograd.grad(kl_grad_vec, initial_params.values(), retain_graph=True, allow_unused=True) # type: ignore
        fvp_flat = torch.cat([g.flatten() for g in fvp if g is not None]).detach()
        return fvp_flat + cg_damping * vec

    x = conjugate_gradient_jvp(fisher_vector_product, obj_grad_flat, max_iter=10, tol=1e-5)
    natural_pg_flat = torch.sqrt(2 * target_kl / (x @ fisher_vector_product(x))) * x
    natural_pg = [torch.zeros_like(p) for p in model.parameters()]
    torch.nn.utils.vector_to_parameters(natural_pg_flat, [p for p,u in zip(natural_pg,used_params) if u]) # type: ignore

    # Clear memory
    del obj, obj_grad, obj_grad_flat, kl_grad, kl_grad_flat, kl, x, natural_pg_flat

    # Line search
    weights_updated = False
    for i in range(10):
        alpha = 0.5 ** i
        new_params = [
                p + alpha * p_natural_pg
                for p, p_natural_pg in zip(model.parameters(), natural_pg)
        ]
        try:
            new_kl = kl_divergence(dict(zip(key_order, new_params)))
            new_kl = new_kl.mean()
            new_loss = -surrogate_objective_dict([
                adapt_model(
                    model = model,
                    params = dict(zip(key_order, new_params)),
                    history = history,
                    discount = discount,
                    gae_lambda = gae_lambda,
                    inner_lr = inner_lr,
                )
                for history,_ in data
            ]).detach().item()
        except Exception as e:
            # The adaptation step might diverge and crash. If it does, we don't use these weights
            print(f'{i} Errored: {e}')
            continue
        if new_kl < target_kl and new_loss < old_loss:
            model.load_state_dict(dict(zip(key_order, new_params)))
            print(f'{i}: Successful update')
            weights_updated = True
            if wandb.run is not None:
                output = {
                    **output,
                    'kl': new_kl,
                    'new_loss': new_loss,
                    'old_loss': old_loss,
                    'ls_step': i,
                }
            break
        else:
            if new_loss > old_loss and new_kl > target_kl:
                print(f'{i}: Both loss increased ({new_loss-old_loss}) and KL too high ({new_kl})')
            elif new_loss > old_loss:
                print(f'{i}: Loss increased ({new_loss-old_loss})')
            elif new_kl > target_kl:
                print(f'{i}: KL too high ({new_kl})')
            else:
                print(f'{i}: Unknown error')

    global doot
    doot[0] += weights_updated
    doot[1] += 1
    print(f'{doot[0]/doot[1]} ({doot[0]}/{doot[1]})')
    if not weights_updated:
        print("No update applied")
    else:
        output = {
            **output,
            **evaluate_v2(model, data)
        }

    output['time'] = time.time() - start_time
    print(f'Time: {output["time"]}')

    return output


def train_trpo_mujoco(
        model: torch.nn.Module,
        envs: list[VectorEnv],
        *,
        inner_lr: float = 0.1,
        max_steps: int = 1000,
        rollout_length: int = 128,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        target_kl: float = 0.01,
        meta_batch_size: int = 20,
        ):
    start_time = time.time()

    env_steps = 0
    for step in itertools.count():
        data = []
        initial_params = dict(model.named_parameters())
        adapted_model_params = []
        for _ in range(meta_batch_size):
            env = random.choice(envs)
            # Generate training data
            train_data = gather_data(
                model = model,
                params = initial_params, # type: ignore
                env = env,
                rollout_length = rollout_length,
            )
            # Adapt model to the training data
            new_model_params = adapt_model(
                model = model,
                params = initial_params, # type: ignore
                history = train_data,
                discount = discount,
                gae_lambda = gae_lambda,
                inner_lr = inner_lr,
            )
            # Generate data from the adapted model
            validation_data = gather_data(
                model = model,
                params = new_model_params,
                env = env,
                rollout_length = rollout_length,
            )
            # Save data for training
            data.append((train_data, validation_data))
            adapted_model_params.append(new_model_params)

        # Train
        print('=' * 80)
        print('Training...')
        print('=' * 80)

        output = update_trpo(
                data = data,
                model = model,
                initial_params = initial_params, # type: ignore
                adapted_params = adapted_model_params,
                inner_lr = inner_lr,
                discount = discount,
                gae_lambda = gae_lambda,
                target_kl = target_kl,
        )

        # Timing
        if step > 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = env_steps / elapsed_time
            if max_steps > 0:
                remaining_time = int((max_steps - env_steps) / steps_per_sec)
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                remaining_seconds = (remaining_time % 3600) % 60
                print(f"Iteration {step:,}/{max_steps:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
            else:
                elapsed_time = int(elapsed_time)
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                elapsed_seconds = (elapsed_time % 3600) % 60
                print(f"Iteration {step:,} \t {int(steps_per_sec):,} SPS \t Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")

        if max_steps > 0 and env_steps >= max_steps:
            print('Max steps reached')
            elapsed_time = time.time() - start_time
            elapsed_time = int(elapsed_time)
            elapsed_hours = elapsed_time // 3600
            elapsed_minutes = (elapsed_time % 3600) // 60
            elapsed_seconds = (elapsed_time % 3600) % 60
            print(f'Run time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')
            break
        
        yield output


def gather_data(model: torch.nn.Module,
                params: dict[str, torch.Tensor],
                env: VectorEnv,
                rollout_length: int
        ) -> VecHistoryBuffer:
    num_envs = env.num_envs
    device = next(model.parameters()).device

    history = VecHistoryBuffer(
            num_envs = num_envs,
            max_len=rollout_length+1,
            device=device)

    obs, info = env.reset()
    with torch.no_grad(): # Just running this to get the shape of the action
        model_output = functional_call(model, params, torch.tensor(obs, dtype=torch.float, device=device))
        action_mean = model_output['action_mean']
    history.append_obs(
            obs,
            misc={
                'task_id': info['target_direction'],
                'log_action_prob': np.ones(num_envs)*np.nan,
                'action_logstd': np.ones_like(action_mean.cpu())*np.nan,
                'action_mean': np.ones_like(action_mean.cpu())*np.nan,
                'time': np.zeros(num_envs),
            }
    )
    for t in range(rollout_length):
        # Select action
        with torch.no_grad():
            model_output = functional_call(model, params, torch.tensor(obs, dtype=torch.float, device=device))
            action_mean = model_output['action_mean']
            action_logstd = model_output['action_logstd']
            action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
            action = action_dist.sample().cpu().numpy()
            log_action_prob = action_dist.log_prob(torch.tensor(action, dtype=torch.float, device=device)).sum(dim=1)
        history.append_action(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action) # type: ignore
        done = terminated | truncated
        history.append_obs(
                obs, reward, done,
                misc={
                    'task_id': info['target_direction'],
                    'log_action_prob': log_action_prob.cpu().numpy(),
                    'action_logstd': action_logstd.cpu().numpy(),
                    'action_mean': action_mean.cpu().numpy(),
                    'time': np.ones(num_envs)*(t+1),
                }
        )

    return history


def adapt_model(model: torch.nn.Module,
                params: dict[str, torch.Tensor],
                history: VecHistoryBuffer,
                *,
                discount: float = 0.99,
                gae_lambda: float = 0.95,
                inner_lr: float = 0.1) -> dict[str, torch.Tensor]:
    key_order = [k for k,_ in model.named_parameters()]
    device = next(model.parameters()).device

    n = len(history.obs_history)
    obs = torch.tensor(history.obs, dtype=torch.float, device=device)
    action = history.action
    reward = history.reward
    terminal = history.terminal
    timesteps = history.misc['time'] # type: ignore
    old_log_action_probs = torch.tensor(history.misc['log_action_prob'][1:], dtype=torch.float, device=device) # type: ignore

    net_output = functional_call(model, params, obs)
    #state_values = net_output['value'].squeeze(2)
    action_mean = net_output['action_mean'][:n-1]
    action_logstd = net_output['action_logstd'][:n-1]
    action_dist = torch.distributions.Normal(action_mean, torch.exp(action_logstd))
    log_action_probs = action_dist.log_prob(action).sum(dim=-1)

    with torch.no_grad():
        # Advantage
        advantage = compute_advantage(
            obs = obs,
            timesteps = timesteps,
            reward = reward,
            terminal = terminal,
            discount = discount,
            gae_lambda = gae_lambda,
        )

    losses = adaptation_loss(
            log_action_probs = log_action_probs,
            old_log_action_probs = old_log_action_probs.detach(),
            terminals = terminal[:n-1,:],
            returns = advantage,
    ).mean()

    grads = torch.autograd.grad(
            losses,
            [params[k] for k in key_order],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
    )

    new_params = {
        k: params[k] - inner_lr * g
        for k,g in zip(key_order, grads)
    }

    return new_params


def adaptation_loss(log_action_probs,
                     old_log_action_probs,
                     terminals,
                     returns):
    return -(torch.exp(log_action_probs - old_log_action_probs) * returns * terminals.logical_not())


def evaluate(
        model: torch.nn.Module,
        env: VectorEnv,
        *,
        learning_rate: float = 0.1,
        rollout_length: int = 200,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        ):
    model = copy.deepcopy(model)
    device = next(model.parameters()).device

    num_envs = env.num_envs

    def finetune(model):
        yield model

        for _ in itertools.count():
            history = gather_data(
                model = model,
                params = dict(model.named_parameters()),
                env = env,
                rollout_length = rollout_length,
            )


            # Adapt model to the training data
            new_model_params = adapt_model(
                model = model,
                params = dict(model.named_parameters()),
                history = history,
                discount = discount,
                gae_lambda = gae_lambda,
                inner_lr = learning_rate,
            )

            # Update model parameters
            model.load_state_dict(new_model_params)

            yield model


    # Evaluate after each fine-tuning step
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    output = {'reward': [], 'length': []}
    for _, m in zip(range(3), finetune(model)):
        try:
            dones = np.zeros(num_envs, dtype=bool)
            obs, _ = env.reset()

            with torch.no_grad():
                for _ in tqdm(itertools.count()):
                    model_output = m(torch.tensor(obs, dtype=torch.float, device=device))
                    action_mean = model_output['action_mean']
                    action_logstd = model_output['action_logstd']
                    action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
                    action = action_dist.sample().cpu().numpy()

                    # Step environment
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated | truncated

                    episode_reward += reward * (1 - dones)
                    episode_steps += (1 - dones)
                    dones |= done

                    if dones.all():
                        break
        except:
            episode_reward[:] = np.nan
            episode_steps[:] = np.nan

        output['reward'].append(episode_reward.copy())
        output['length'].append(episode_steps.copy())

        episode_reward *= 0
        episode_steps *= 0

    return output


def evaluate_v2(
        model: torch.nn.Module,
        history: list[tuple[VecHistoryBuffer, VecHistoryBuffer]],
        *,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        ):
    device = next(model.parameters()).device

    history_by_task = {-1: [], 1: []}
    for h in history:
        history_by_task[h[0].misc['task_id'][0,0].item()].append(h[0]) # type: ignore

    def make_env(name):
        env = gym.make(name)
        if args.episode_length is not None:
            env = TimeLimit(env, max_episode_steps=args.episode_length)

        env = RecordEpisodeStatistics(env)

        env = ClipAction(env)

        #env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        #env = NormalizeReward(env, gamma=args.discount)
        env = TransformReward(env, lambda reward: np.clip(reward * args.reward_scale, -args.reward_clip, args.reward_clip))
        return env

    def finetune(model, task_id):
        yield model

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        device = next(model.parameters()).device
        for _ in itertools.count():
            # Train
            n = len(history_by_task[task_id][0].obs_history)
            obs = torch.tensor(history_by_task[task_id][0].obs, dtype=torch.float, device=device)
            action = history_by_task[task_id][0].action
            reward = history_by_task[task_id][0].reward
            terminal = history_by_task[task_id][0].terminal
            device = next(model.parameters()).device
            task_ids = history_by_task[task_id][0].misc['task_id'][0,:] # type: ignore
            indices = torch.arange(task_ids.shape[0], device=device)[task_ids == task_id]
            timesteps = history_by_task[task_id][0].misc['time'] # type: ignore

            net_output = model(obs)
            action_mean = net_output['action_mean'][:n-1]
            action_logstd = net_output['action_logstd'][:n-1]
            action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
            log_action_probs = action_dist.log_prob(action).sum(-1)

            # Advantage
            with torch.no_grad():
                advantage = compute_advantage(
                    obs = obs.index_select(1, indices),
                    timesteps = timesteps.index_select(1, indices),
                    reward = reward.index_select(1, indices),
                    terminal = terminal.index_select(1, indices),
                    discount = discount,
                    gae_lambda = gae_lambda,
                )

            loss = adaptation_loss(
                log_action_probs=log_action_probs.index_select(1, indices),
                old_log_action_probs=log_action_probs.detach().index_select(1, indices),
                terminals=terminal[:n-1,:].index_select(1, indices),
                returns=advantage,
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yield model

    num_envs = 5
    envs = {}
    envs[1] = SyncVectorEnv([lambda: make_env('HalfCheetahForward-v4') for _ in range(num_envs)])
    envs[-1] = SyncVectorEnv([lambda: make_env('HalfCheetahBackward-v4') for _ in range(num_envs)])

    # Evaluate after each fine-tuning step
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    output = {}
    for direction in [-1, 1]:
        print('Direction:', direction)
        env = envs[direction]
        for ep, m in zip(range(2), finetune(copy.deepcopy(model), direction)):
            try:
                dones = np.zeros(num_envs, dtype=bool)
                obs, _ = env.reset()

                with torch.no_grad():
                    for _ in itertools.count():
                        model_output = m(torch.tensor(obs, dtype=torch.float, device=device))
                        action_mean = model_output['action_mean']
                        action_logstd = model_output['action_logstd']
                        action_dist = torch.distributions.Normal(action_mean, action_logstd.exp())
                        action = action_dist.sample().cpu().numpy()

                        # Step environment
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated | truncated

                        episode_reward += reward * (1 - dones)
                        episode_steps += (1 - dones)
                        dones |= done

                        if dones.all():
                            break
            except:
                episode_reward[:] = np.nan
                episode_steps[:] = np.nan

            if ep == 0:
                print('  Before fine-tuning')
            else:
                print('  After fine-tuning')
            print('    Reward:', episode_reward)
            print('    Reward mean:', np.mean(episode_reward))

            #output['reward'].append(episode_reward.copy())
            #output['length'].append(episode_steps.copy())

            if wandb.run is not None and ep == 1:
                #wandb.log({f'adaptation_reward_{"right" if direction == 1 else "left"}': np.mean(episode_reward)})
                output[f'adaptation_reward_{"right" if direction == 1 else "left"}'] = np.mean(episode_reward)

            episode_reward *= 0
            episode_steps *= 0

    return output


def init_arg_parser():
    N_ENVS = 20
    HORIZON = 200

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetahForward-v4', help='Environment to train on')
    parser.add_argument('--episode-length', type=int, default=200, help='Length of an episode. If it exceeds the default time limit, then the default will take precedence.')
    parser.add_argument('--num-envs', type=int, default=N_ENVS, help='Number of environments to train on')
    parser.add_argument('--max-steps', type=int, default=200_000_000//HORIZON//N_ENVS, help='Number of training steps to run. One step is one weight update. If 0, train forever.')
    parser.add_argument('--inner-lr', type=float, default=0.1, help='Learning rate for the fine-tuning step.')
    parser.add_argument('--rollout-length', type=int, default=HORIZON, help='Length of rollout.')
    parser.add_argument('--reward-clip', type=float, default=10, help='Clip the reward magnitude to this value.') # CleanRL uses 10
    parser.add_argument('--reward-scale', type=float, default=1, help='Scale the reward magnitude by this value.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for GAE.')
    parser.add_argument('--target-kl', type=float,
                        default=0.01, # See table 2 of TRPO paper (https://arxiv.org/pdf/1502.05477.pdf). This is listed as the step size.
                        help='Target KL divergence.')
    parser.add_argument('--meta-batch-size', type=int,
                        default=20,
                        help='')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases.')

    return parser


def main(args):
    # Validate args
    if args.rollout_length % args.episode_length != 0:
        raise ValueError('Rollout length must be a multiple of episode length')

    if args.wandb:
        wandb.init(project='maml-cheetah')
        wandb.config.update(args)

    def make_env(name):
        env = gym.make(name)
        if args.episode_length is not None:
            env = TimeLimit(env, max_episode_steps=args.episode_length)

        env = RecordEpisodeStatistics(env)

        env = ClipAction(env)

        #env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        #env = NormalizeReward(env, gamma=args.discount)
        env = TransformReward(env, lambda reward: np.clip(reward * args.reward_scale, -args.reward_clip, args.reward_clip))
        return env
    envs = [
        AsyncVectorEnv([lambda: make_env('HalfCheetahForward-v4') for _ in range(args.num_envs)]),
        AsyncVectorEnv([lambda: make_env('HalfCheetahBackward-v4') for _ in range(args.num_envs)]),
    ]
    num_test_envs = 20
    test_env_forward = AsyncVectorEnv([lambda: make_env('HalfCheetahForward-v4') for _ in range(num_test_envs)])
    test_env_backward = AsyncVectorEnv([lambda: make_env('HalfCheetahBackward-v4') for _ in range(num_test_envs)])

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    assert envs[0].single_observation_space.shape is not None
    assert envs[0].single_action_space.shape is not None
    model = Model4(
            obs_size=envs[0].single_observation_space.shape[0],
            action_size=envs[0].single_action_space.shape[0]
    )
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {param_count:,}')

    trainer = train_trpo_mujoco(
            model = model,
            envs = envs, # type: ignore
            inner_lr = args.inner_lr,
            max_steps = args.max_steps,
            rollout_length = args.rollout_length,
            discount = args.discount,
            gae_lambda = args.gae_lambda,
            target_kl = args.target_kl,
    )
    #episode_rewards = []
    #episode_lengths = []
    test_rewards_forward = []
    test_lengths_forward = []
    test_rewards_backward = []
    test_lengths_backward = []
    for i in itertools.count():
        if i % 10 == 0:
            performance_forward = evaluate(model, test_env_forward, learning_rate=args.inner_lr, rollout_length=args.rollout_length)
            performance_backward = evaluate(model, test_env_backward, learning_rate=args.inner_lr, rollout_length=args.rollout_length)
            test_rewards_forward.append(performance_forward['reward'])
            test_lengths_forward.append(performance_forward['length'])
            test_rewards_backward.append(performance_backward['reward'])
            test_lengths_backward.append(performance_backward['length'])
            print(textwrap.dedent(f"""\
                Step {i}:
                  reward_forward: {performance_forward['reward']}
                  length_forward: {performance_forward['length']}
                  reward_backward: {performance_backward['reward']}
                  length_backward: {performance_backward['length']}
            """.strip('\n')))
            if wandb.run is not None:
                data = {}
                for grad_steps in range(3):
                    data[f'test_performance/forward/{grad_steps}'] = performance_forward['reward'][grad_steps]
                    data[f'test_performance/backward/{grad_steps}'] = performance_backward['reward'][grad_steps]
                wandb.log(data, step=i)
        #if i % 100 == 0:
        #    checkpoint_directory = 'checkpoints-2048'
        #    os.makedirs(checkpoint_directory, exist_ok=True)
        #    torch.save(model.state_dict(), f'{checkpoint_directory}/checkpoint_{i}.pt')
        x = next(trainer)
        if wandb.run is not None:
            wandb.log(x, step=i)
        #episode_rewards.extend(x['reward'])
        #episode_lengths.extend(x['length'])

    objective_forward = np.mean([x[1] for x in test_rewards_forward])
    objective_backward = np.mean([x[1] for x in test_rewards_backward])
    objective = min(objective_forward, objective_backward) # type: ignore
    if args.wandb:
        wandb.log({'objective': objective})


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)

