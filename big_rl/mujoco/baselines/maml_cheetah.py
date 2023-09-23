import argparse
import copy
import itertools
import time
import textwrap

import gymnasium as gym
from gymnasium.vector import VectorEnv, AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, NormalizeObservation, TransformObservation, NormalizeReward, TransformReward, TimeLimit # pyright: ignore[reportPrivateImportUsage]
import torch
from torch.func import functional_call # pyright: ignore[reportPrivateImportUsage]
from torch.utils.data.dataloader import default_collate
import numpy as np
import wandb
from tqdm import tqdm

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
import big_rl.mujoco.envs


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
        batch_size = x.shape[0]
        return {
                'value': self.v(x),
                #'action': pi,
                'action_mean': self.pi_mean(x),
                'action_logstd': self.pi_logstd[None, :].expand(batch_size, -1),
        }


def update_trpo(
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        discount : float,
        gae_lambda : float,
        target_kl : float):
    obs = history.obs
    action = history.action
    reward = history.reward
    terminal = history.terminal
    device = next(model.parameters()).device

    n = len(history.obs_history)

    key_order = [k for k,_ in model.named_parameters()]

    def model_fn_dict(params_dict):
        return functional_call(model, params_dict, torch.tensor(obs, dtype=torch.float, device=device))

    with torch.no_grad():
        net_output = model_fn_dict(dict(model.named_parameters()))
        state_values_old = net_output['value'].squeeze(2)
        action_mean_old = net_output['action_mean'][:n-1]
        action_logstd_old = net_output['action_logstd'][:n-1]

        # Advantage
        advantages = generalized_advantage_estimate(
                state_values = state_values_old[:n-1,:],
                next_state_values = state_values_old[1:,:],
                rewards = reward[1:,:],
                terminals = terminal[1:,:],
                discount = discount,
                gae_lambda = gae_lambda,
        )
        returns = advantages + state_values_old[:n-1,:]

    # Update value function
    net_output = model_fn_dict(dict(model.named_parameters()))
    state_values = net_output['value'].squeeze(2)
    value_loss = (state_values[:n-1] - returns).pow(2).mean()

    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()

    with torch.no_grad():
        net_output = model_fn_dict(dict(model.named_parameters()))
        state_values_old = net_output['value'].squeeze(2)
        action_mean_old = net_output['action_mean'][:n-1]
        action_logstd_old = net_output['action_logstd'][:n-1]

        # Advantage
        advantages = generalized_advantage_estimate(
                state_values = state_values_old[:n-1,:],
                next_state_values = state_values_old[1:,:],
                rewards = reward[1:,:],
                terminals = terminal[1:,:],
                discount = discount,
                gae_lambda = gae_lambda,
        )

    def surrogate_objective_dict(model_params_dict):
        inner_lr = 0.1

        net_output = model_fn_dict(model_params_dict)
        #state_values = net_output['value'].squeeze(2)
        action_mean = net_output['action_mean'][:n-1]
        action_logstd = net_output['action_logstd'][:n-1]
        action_dist = torch.distributions.Normal(action_mean, action_logstd.exp()+1e-5)
        log_action_probs = action_dist.log_prob(action).sum(-1)
        losses = -(log_action_probs * advantages).mean(0)

        grads = [
                torch.autograd.grad(loss, [model_params_dict[k] for k in key_order], create_graph=True, allow_unused=True)
                for loss in losses
        ]
        new_model_params = [
            {
                k: model_params_dict[k] - inner_lr * (g if g is not None else 0)
                for k,g in zip(key_order, grad)
            }
            for grad in grads
        ]

        net_output_list = [
                functional_call(model, p, torch.tensor(o, dtype=torch.float, device=device))
                for p,o in zip(new_model_params, obs.split(1, dim=1))
        ]
        net_output = {
            k: torch.cat([o[k] for o in net_output_list], dim=1)
            for k in net_output_list[0].keys()
        }

        action_mean = net_output['action_mean'][:n-1]
        action_logstd = net_output['action_logstd'][:n-1]
        action_dist = torch.distributions.Normal(action_mean, action_logstd.exp()+1e-5)
        log_action_probs = action_dist.log_prob(action).sum(-1)
        return (log_action_probs * advantages).mean()

    
    def kl_divergence(model_params_dict):
        net_output = model_fn_dict(model_params_dict)
        action_mean = net_output['action_mean'][:n-1]
        action_logstd = net_output['action_logstd'][:n-1]
        # See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        # For the KL divergence between two normal distributions
        kl = action_logstd - action_logstd_old + (action_logstd_old.exp()**2 + (action_mean_old - action_mean)**2)/(2*action_logstd.exp()**2) - 0.5
        return kl

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

    params = dict(model.named_parameters())
    obj = surrogate_objective_dict(params)
    obj_grad = torch.autograd.grad(obj, params.values(), create_graph=True, allow_unused=True) # type: ignore
    obj_grad_flat = torch.cat([g.flatten() for g in obj_grad if g is not None]).detach()
    used_params = [g is not None for g in obj_grad]
    kl = kl_divergence(params)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, params.values(), create_graph=True, allow_unused=True) # type: ignore
    kl_grad_flat = torch.cat([g.flatten() for g in kl_grad if g is not None])
    def fisher_vector_product(vec):
        kl_grad_vec = kl_grad_flat @ vec
        fvp = torch.autograd.grad(kl_grad_vec, params.values(), retain_graph=True, allow_unused=True) # type: ignore
        fvp_flat = torch.cat([g.flatten() for g in fvp if g is not None]).detach()
        return fvp_flat #+ cg_damping * vec # What is this damping thing?

    x = conjugate_gradient_jvp(fisher_vector_product, obj_grad_flat, max_iter=100, tol=1e-5)
    natural_pg_flat = torch.sqrt(2 * target_kl / (x @ fisher_vector_product(x))) * x
    natural_pg = [torch.zeros_like(p) for p in model.parameters()]
    torch.nn.utils.vector_to_parameters(natural_pg_flat, [p for p,u in zip(natural_pg,used_params) if u]) # type: ignore

    # Line search
    new_kl = torch.tensor(0)
    for i in range(10):
        alpha = 0.5 ** i
        new_params = [
                p + alpha * p_natural_pg
                for p, p_natural_pg in zip(model.parameters(), natural_pg)
        ]
        new_kl = kl_divergence(dict(zip(key_order, new_params)))
        new_kl = new_kl.mean()
        if new_kl < target_kl:
            model.load_state_dict(dict(zip(key_order, new_params)))
            break


def train_trpo_mujoco(
        model: torch.nn.Module,
        env: VectorEnv,
        optimizer: torch.optim.Optimizer,
        *,
        max_steps: int = 1000,
        rollout_length: int = 128,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        target_kl: float = 0.01,
        ):
    num_envs = env.num_envs
    device = next(model.parameters()).device

    history = VecHistoryBuffer(
            num_envs = num_envs,
            max_len=rollout_length+1,
            device=device)
    start_time = time.time()

    obs, _ = env.reset()
    history.append_obs(obs)
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    output = {'reward': [], 'length': []}
    env_steps = 0
    for step in itertools.count():
        # Gather data
        for i in range(rollout_length):
            env_steps += num_envs

            # Select action
            with torch.no_grad():
                model_output = model(torch.tensor(obs, dtype=torch.float, device=device))
                action_mean = model_output['action_mean']
                action_logstd = model_output['action_logstd']
                action_dist = torch.distributions.Normal(action_mean, action_logstd.exp()+1e-5)
                action = action_dist.sample().cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated | truncated

            history.append_action(action)
            episode_reward += reward
            episode_steps += 1

            history.append_obs(obs, reward, done)

            if done.any():
                ep_rew = np.array([x['episode']['r'] for x in info['final_info'] if x is not None])
                ep_len = np.array([x['episode']['l'] for x in info['final_info'] if x is not None])
                print(f'{step * num_envs * rollout_length:,} ({env_steps:,})\t reward: {ep_rew.mean():.2f}\t len: {ep_len.mean()} \t ({done.sum()} done)')
                if wandb.run is not None:
                    wandb.log({
                            'reward': ep_rew.mean().item(),
                            'episode_length': ep_len.mean().item(),
                            'step': env_steps,
                    }, step = env_steps)
                episode_reward[done] = 0
                episode_steps[done] = 0

                for x in info['final_info']:
                    if x is None:
                        continue
                    output['reward'].append(x['episode']['r'])
                    output['length'].append(x['episode']['l'])

        # Train
        print('=' * 80)
        print('Training...')
        print('=' * 80)
        update_trpo(
                history = history,
                model = model,
                optimizer = optimizer,
                discount = discount,
                gae_lambda = gae_lambda,
                target_kl = target_kl,
        )

        # Clear data
        history.clear()

        # Timing
        if step > 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = env_steps / elapsed_time
            if max_steps > 0:
                remaining_time = int((max_steps - env_steps) / steps_per_sec)
                remaining_hours = remaining_time // 3600
                remaining_minutes = (remaining_time % 3600) // 60
                remaining_seconds = (remaining_time % 3600) % 60
                print(f"Step {env_steps:,}/{max_steps:,} \t {int(steps_per_sec):,} SPS \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")
            else:
                elapsed_time = int(elapsed_time)
                elapsed_hours = elapsed_time // 3600
                elapsed_minutes = (elapsed_time % 3600) // 60
                elapsed_seconds = (elapsed_time % 3600) % 60
                print(f"Step {env_steps:,} \t {int(steps_per_sec):,} SPS \t Elapsed: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")

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
        output = {'reward': [], 'length': []}


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

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        device = next(model.parameters()).device
        for _ in itertools.count():
            history = VecHistoryBuffer(
                    num_envs = num_envs,
                    max_len=rollout_length+1,
                    device=device)

            obs, _ = env.reset()
            history.append_obs(obs)

            # Gather data
            for _ in range(rollout_length):
                # Select action
                with torch.no_grad():
                    model_output = model(torch.tensor(obs, dtype=torch.float, device=device))
                    action_mean = model_output['action_mean']
                    action_logstd = model_output['action_logstd']
                    action_dist = torch.distributions.Normal(action_mean, action_logstd.exp()+1e-5)
                    action = action_dist.sample().cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated

                history.append_action(action)
                history.append_obs(obs, reward, done)

            # Train
            obs = history.obs
            action = history.action
            reward = history.reward
            terminal = history.terminal
            device = next(model.parameters()).device

            n = len(history.obs_history)

            net_output = model(torch.tensor(obs, dtype=torch.float, device=device))
            state_values = net_output['value'].squeeze(2)
            action_mean = net_output['action_mean'][:n-1]
            action_logstd = net_output['action_logstd'][:n-1]
            action_dist = torch.distributions.Normal(action_mean, action_logstd.exp()+1e-5)
            log_action_probs = action_dist.log_prob(action).sum(-1)

            # Advantage
            with torch.no_grad():
                advantages = generalized_advantage_estimate(
                        state_values = state_values[:n-1,:],
                        next_state_values = state_values[1:,:],
                        rewards = reward[1:,:],
                        terminals = terminal[1:,:],
                        discount = discount,
                        gae_lambda = gae_lambda,
                )

            loss = -(log_action_probs * advantages).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yield model

            # Clear data
            history.clear()

    # Evaluate after each fine-tuning step
    episode_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs)
    output = {'reward': [], 'length': []}
    for _, model in zip(range(3), finetune(model)):
        dones = np.zeros(num_envs, dtype=bool)
        obs, _ = env.reset()

        with torch.no_grad():
            for _ in tqdm(itertools.count()):
                model_output = model(torch.tensor(obs, dtype=torch.float, device=device))
                action_mean = model_output['action_mean']
                action_logstd = model_output['action_logstd']
                action_dist = torch.distributions.Normal(action_mean, action_logstd.exp()+1e-5)
                action = action_dist.sample().cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated

                episode_reward += reward * (1 - dones)
                episode_steps += (1 - dones)
                dones |= done

                if dones.all():
                    break

        output['reward'].append(episode_reward.copy())
        output['length'].append(episode_steps.copy())

        episode_reward *= 0
        episode_steps *= 0

    return output


def init_arg_parser():
    N_ENVS = 20
    HORIZON = 200

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='HalfCheetah-v4', help='Environment to train on')
    parser.add_argument('--episode-length', type=int, default=200, help='Length of an episode. If it exceeds the default time limit, then the default will take precedence.')
    parser.add_argument('--num-envs', type=int, default=N_ENVS, help='Number of environments to train on')
    parser.add_argument('--max-steps', type=int, default=200_000_000//HORIZON//N_ENVS, help='Number of training steps to run. One step is one weight update. If 0, train forever.')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer', choices=['SGD', 'Adam', 'RMSprop'])
    parser.add_argument('--lr', type=float, default=3.0e-4, help='Baseline value function learning rate.')
    parser.add_argument('--inner-lr', type=float, default=3.0e-4, help='Learning rate for the fine-tuning step.')
    parser.add_argument('--rollout-length', type=int, default=HORIZON, help='Length of rollout.')
    parser.add_argument('--reward-clip', type=float, default=10, help='Clip the reward magnitude to this value.') # CleanRL uses 10
    parser.add_argument('--reward-scale', type=float, default=1, help='Scale the reward magnitude by this value.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for GAE.')
    parser.add_argument('--target-kl', type=float,
                        default=0.01, # See table 2 of TRPO paper (https://arxiv.org/pdf/1502.05477.pdf). This is listed as the step size.
                        help='Target KL divergence.')

    parser.add_argument('--cuda', action='store_true', help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true', help='Use Weights and Biases.')

    return parser


def main(args):
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
    env = AsyncVectorEnv([lambda: make_env(args.env) for _ in range(args.num_envs)])
    #test_env = AsyncVectorEnv([lambda: make_env(args.env) for _ in range(2)])
    test_env_forward = AsyncVectorEnv([lambda: make_env('HalfCheetahForward-v4') for _ in range(1)])
    test_env_backward = AsyncVectorEnv([lambda: make_env('HalfCheetahBackward-v4') for _ in range(1)])

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    assert env.single_observation_space.shape is not None
    assert env.single_action_space.shape is not None
    model = Model(
            obs_size=env.single_observation_space.shape[0],
            action_size=env.single_action_space.shape[0]
    )
    model.to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    trainer = train_trpo_mujoco(
            model = model,
            env = env,
            optimizer = optimizer,
            max_steps = args.max_steps,
            rollout_length = args.rollout_length,
            discount = args.discount,
            gae_lambda = args.gae_lambda,
            target_kl = args.target_kl,
    )
    episode_rewards = []
    episode_lengths = []
    test_rewards_forward = []
    test_lengths_forward = []
    test_rewards_backward = []
    test_lengths_backward = []
    for i,x in enumerate(trainer):
        if i % 10 == 0:
            performance_forward = evaluate(model, test_env_forward)
            performance_backward = evaluate(model, test_env_backward)
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
                wandb.log(data)
        episode_rewards.extend(x['reward'])
        episode_lengths.extend(x['length'])

    objective_forward = np.mean([x[1] for x in test_rewards_forward])
    objective_backward = np.mean([x[1] for x in test_rewards_backward])
    objective = min(objective_forward, objective_backward) # type: ignore
    if args.wandb:
        wandb.log({'objective': objective})


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)

