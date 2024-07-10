from collections import defaultdict
import itertools
import os
from typing import Any, Generator, Iterable
import uuid
import yaml

import numpy as np
import torch
import torch.utils.data
import gymnasium
from tensordict import TensorDict
from tqdm import tqdm
import wandb

import big_rl.utils.args
from big_rl.utils import generate_id
from big_rl.utils.make_env import make_env_from_yaml
from big_rl.utils import merge_space
from big_rl.model import factory as model_factory
from big_rl.generic.evaluate_model import main as evaluate_model_main_, init_arg_parser as evaluate_model_init_arg_parser_


WANDB_PROJECT_NAME = 'distillation'


def cache_to_file(func, filename):
    if os.path.exists(filename):
        return torch.load(filename)
    output = func()
    torch.save(output, filename)
    return output


def action_dist_discrete(net_output, n=None):
    return torch.distributions.Categorical(logits=net_output['action'][:n])


def action_dist_continuous(net_output, n=None):
    action_mean = net_output['action_mean'][:n]
    action_logstd = net_output['action_logstd'][:n]
    return torch.distributions.Normal(action_mean, action_logstd.exp())


def get_action_dist_function(action_space: gymnasium.Space):
    if isinstance(action_space, gymnasium.spaces.Discrete):
        return action_dist_discrete
    elif isinstance(action_space, gymnasium.spaces.Box):
        return action_dist_continuous
    else:
        raise NotImplementedError(f'Unknown action space: {action_space}')


def generate_experience(model, env: gymnasium.vector.VectorEnv, seed=None, device=torch.device('cpu'), num_episodes=1000) -> list[list[tuple[Any, Any]]]:
    action_dist_fn = get_action_dist_function(env.single_action_space)
    num_envs = env.num_envs
    episodes = [[] for _ in range(num_envs)] # For accumulating the episode data before saving it to the dataset
    dataset = [] # List of completed episodes

    obs, _ = env.reset(seed=seed)
    hidden = model.init_hidden(num_envs) # type: ignore (???)

    for step in tqdm(itertools.count(), desc='Generating experience'):
        # Select action
        with torch.no_grad():
            model_output = model({
                k: torch.tensor(v, dtype=torch.float, device=device)
                for k,v in obs.items()
            }, hidden)
            hidden = model_output['hidden']

            action_dist = action_dist_fn(model_output)
            action = action_dist.sample().cpu().numpy()

        # Save obs and action dist
        obs['obs'] = obs['obs'].astype(np.float32)
        td_obs = TensorDict(obs, batch_size=num_envs)
        for i in range(num_envs):
            if isinstance(action_dist, torch.distributions.Normal):
                episodes[i].append((
                    td_obs[i],
                    torch.stack([action_dist.loc[i], action_dist.scale[i]]),
                ))
            elif isinstance(action_dist, torch.distributions.Categorical):
                episodes[i].append((
                    td_obs[i],
                    action_dist.logits[i], # type: ignore
                ))
            else:
                raise NotImplementedError(f'Unknown action distribution: {action_dist}')

        # Step environment
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # Save to dataset
        for i in range(num_envs):
            if done[i]:
                dataset.append(episodes[i])
                episodes[i] = []

        if done.any():
            # Reset hidden state for finished episodes
            hidden = tuple(
                    torch.where(torch.tensor(done, device=device).unsqueeze(1), h0, h)
                    for h0,h in zip(model.init_hidden(num_envs), hidden) # type: ignore (???)
            )

            # Print stats
            #print(f'Reward: {episode_true_reward[done].mean():.2f}\t len: {episode_steps[done].mean()} \t completed envs: {done.sum().item()}')

        # Check if we have enough episodes
        if len(dataset) >= num_episodes:
            break

    return dataset


def shuffle(data, repeat=1, rng: np.random.Generator = np.random.default_rng()):
    for _ in range(repeat):
        indices = np.arange(len(data))
        rng.shuffle(indices)
        for i in indices:
            yield data[i]


def train_single_model(dataset: list[list[tuple[Any, Any]]], model, optimizer, device=torch.device('cpu'), batch_size=1):
    # Given a dataset of episodes, train the model on the dataset
    # This is for recurrent models, so we need to keep track of the hidden state and train on the entire episode at once
    # We also need to handle the fact that the dataset may contain episodes of different lengths
    # Just do one episode at a time for now instead of batching them
    # `dataset` is a list of episodes, where each episode is a list of (obs, action dist) pairs
    # Assume continuous action space for now
    # Only distill the policy. Ignore the value function.
    
    loss_batch = []
    for episode in tqdm(shuffle(dataset), total=len(dataset), desc='Training'):
        hidden = model.init_hidden(1)
        output = []
        for obs, _ in episode:
            model_output = model(obs.unsqueeze(0), hidden)
            hidden = model_output['hidden']
            output.append(model_output)
        output = torch.utils.data.default_collate(output)

        # Batch the distributions
        dist = torch.distributions.Normal(output['action_mean'].squeeze(1), output['action_logstd'].squeeze(1).exp())
        target_dist_batch = torch.stack([d for _,d in episode], dim=0) # [episode len, 2, action dim]
        target_dist = torch.distributions.Normal(target_dist_batch[:,0,:], target_dist_batch[:,1,:])

        # Compute the distance between the teacher and student distributions
        # Use Jensen-Shannon divergence as the distance metric
        jsd = (torch.distributions.kl_divergence(dist, target_dist) + torch.distributions.kl_divergence(target_dist, dist)).mean() / 2

        # Accumulate gradient
        loss_batch.append(jsd)

        # Gradient step
        if len(loss_batch) >= batch_size:
            optimizer.zero_grad()
            loss = torch.stack(loss_batch).mean()
            loss.backward()
            optimizer.step()
            loss_batch = []

            if wandb.run is not None:
                wandb.log({'loss': loss.item()}, commit=True)


def train_multi_model(datasets: list[list[list[tuple[Any, Any]]]], models: list, optimizer, device=torch.device('cpu'), batch_size=1, rng: np.random.Generator = np.random.default_rng()):
    if len(datasets) != len(models):
        raise ValueError('Number of datasets must match the number of models')
    num_models = len(models)
    num_steps = max(len(dataset) for dataset in datasets) # FIXME: Doesn't work if datasets are different sizes

    def step_data_generator() -> Generator[Iterable[tuple[Any, torch.nn.Module]], None, None]:
        generators = []
        for dataset, model in zip(datasets, models):
            if len(dataset) < num_steps:
                # Not handling this because lazy. No need to.
                # Causes problems when one of the datasets is length 0, so when we zip, we get an empty generator
                raise ValueError('Found datasets of different lengths.')
            generators.append(
                ((episode,model) for episode in shuffle(dataset, rng=rng))
            )
        yield from zip(*generators)

    loss_batch = []
    for step_data in tqdm(step_data_generator(), total=num_steps, desc='Training'):
        loss_batch.append([])
        for episode,model in step_data:
            hidden = model.init_hidden(1)
            output = []
            for obs, _ in episode:
                model_output = model(obs.unsqueeze(0), hidden)
                hidden = model_output['hidden']
                output.append(model_output)
            output = torch.utils.data.default_collate(output)

            # Batch the distributions
            dist = torch.distributions.Normal(output['action_mean'].squeeze(1), output['action_logstd'].squeeze(1).exp())
            target_dist_batch = torch.stack([d for _,d in episode], dim=0) # [episode len, 2, action dim]
            target_dist = torch.distributions.Normal(target_dist_batch[:,0,:], target_dist_batch[:,1,:])

            # Compute the distance between the teacher and student distributions
            # Use Jensen-Shannon divergence as the distance metric
            jsd = (torch.distributions.kl_divergence(dist, target_dist) + torch.distributions.kl_divergence(target_dist, dist)).mean() / 2

            # Accumulate gradient
            loss_batch[-1].append(jsd)

        # Gradient step
        if len(loss_batch) >= batch_size:
            optimizer.zero_grad()
            loss = torch.stack([torch.stack(l) for l in loss_batch])
            loss.mean().backward()
            optimizer.step()
            loss_batch = []

            loss = loss.detach().cpu()
            tqdm.write(f'Loss: {loss.mean().item()} {loss.mean(0).numpy().tolist()}')

            if wandb.run is not None:
                log_data = {}
                for i,l in enumerate(loss.mean(0).numpy().tolist()):
                    log_data[f'loss/{i}'] = l
                log_data['loss'] = loss.mean().item()
                wandb.log(log_data, commit=True)


def evaluate(model: torch.nn.Module, env_config, model_config, num_episodes: int, results_filename: str | None, tempdir: str):
    os.makedirs(tempdir, exist_ok=True)
    checkpoint_filename = os.path.join(tempdir, f'model-{uuid.uuid4()}.pt')
    torch.save({'model': model.state_dict()}, checkpoint_filename)

    argparser = evaluate_model_init_arg_parser_()
    args = [
        '--env-config', env_config,
        '--model-config', model_config,
        '--model', checkpoint_filename,
        '--num-episodes', str(num_episodes),
        '--no-video',
    ]
    if results_filename is not None:
        args.extend(['--results', results_filename])
    evaluate_model_main_(argparser.parse_args(args))


def init_teacher_models(args, envs) -> dict[str, torch.nn.Module]:
    output = {}

    if len(args.teacher) > 0:
        # Validation
        if args.teacher_model_config is not None:
            raise ValueError('Cannot specify both --teacher and --teacher_model_config')
        if args.teacher_checkpoint is not None:
            raise ValueError('Cannot specify both --teacher and --teacher_checkpoint')

        # Load teacher models
        for config in args.teacher:
            if len(config) == 2:
                env_name, model_config_filename = config
                checkpoint_filename = None
            elif len(config) == 3:
                env_name, model_config_filename, checkpoint_filename = config
            else:
                raise ValueError(f'Teacher configuration (--teacher) must be specified as a list of two or three values: [ENV_NAME MODEL_CONFIG_YAML [CHECKPOINT_FILE]]. Got: {config}')

            with open(model_config_filename, 'r') as f:
                model_config = yaml.safe_load(f)
            model = model_factory.create_model(
                model_config,
                envs = envs,
            )
            if checkpoint_filename is not None:
                checkpoint = torch.load(checkpoint_filename)
                model.load_state_dict(checkpoint['model'])
            output[env_name] = model
    else:
        with open(args.teacher_model_config, 'r') as f:
            model_configs = yaml.safe_load(f)
        model = model_factory.create_model(
            model_configs,
            envs = envs,
        )

        if args.teacher_checkpoint is not None:
            checkpoint = torch.load(args.teacher_checkpoint)
            model.load_state_dict(checkpoint['model'])

        for env in envs:
            output[env.name] = model

    return output


def init_arg_parser():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-config', type=str, default=None,
                        help='Path to an environment config file (yaml format). If specified, all other environment arguments are ignored.')
    parser.add_argument('--student-model-config', type=str, default=None,
                        help='Path to a model config file (yaml format).')
    parser.add_argument('--teacher-model-config', type=str, default=None,
                        help='Path to a model config file (yaml format).')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                        help='Path to a teacher model checkpoint file.')
    parser.add_argument('--teacher', type=str, action='append', nargs='+',
                        help='Teacher model configuration. Should be specified as a list of two or three values: [ENV_NAME MODEL_CONFIG_YAML [CHECKPOINT_FILE]].')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a student model checkpoint file.')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Number of epochs between saving checkpoints.')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Directory to save results to.')

    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes to generate experience for.')
    parser.add_argument('--epochs-per-dataset', type=int, default=5,
                        help='Number of epochs to use a dataset before generating a new one.')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of episodes to use per gradient step.')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes to run.')

    parser.add_argument('--run-id', type=str, default=None,
                        help='Unique ID for this run.')

    parser.add_argument('--wandb', action='store_true', help='Save results to W&B.')
    parser.add_argument('--wandb-id', type=str, default=None,
                        help='W&B run ID.')
    parser.add_argument('--wandb-project', type=str,
                        default=WANDB_PROJECT_NAME,
                        help='W&B project name.')
    parser.add_argument('--wandb-group', type=str, default='null',
                        help='JSON data used to group runs together in W&B.')

    return parser


##################################################
# Main script
##################################################


def main(args):
    if args.run_id is None:
        args.run_id = generate_id()
    big_rl.utils.args.substitute_vars(args, {**{k:v for k,v in os.environ.items() if k.startswith('SLURM')}, 'RUN_ID': args.run_id})

    if args.wandb:
        if args.wandb_id is None:
            wandb.init(project=args.wandb_project)
        else:
            wandb.init(project=args.wandb_project, id=args.wandb_id, resume='allow')
        wandb.config.update({k:v for k,v in args.__dict__.items() if k != 'model_config'})

    tempdir = os.environ.get('SLURM_TMPDIR', '/tmp')
    tempdir = os.path.join(tempdir, uuid.uuid4().hex)

    envs = make_env_from_yaml(args.env_config)

    # Init student model
    with open(args.student_model_config, 'r') as f:
        student_model_config = yaml.safe_load(f)
    student_model = model_factory.create_model(
        student_model_config,
        envs = envs,
    )

    # Init teacher model(s)
    teacher_models = init_teacher_models(args, envs)

    # Sanity check: Is this a good teacher?
    #print('Evaluating teacher model')
    #evaluate(
    #    model=teacher_models[???],
    #    env_config=args.env_config,
    #    model_config=args.student_model_config,
    #    num_episodes=10,
    #    results_filename=os.path.join(tempdir, 'teacher-results.pt'),
    #    tempdir=tempdir,
    #)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    # Checkpointing
    checkpoint_filename = None
    checkpoint = None
    if args.results_dir is not None:
        checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
        checkpoint_filename = os.path.join(checkpoint_dir, 'checkpoint.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        if os.path.exists(checkpoint_filename):
            checkpoint = torch.load(checkpoint_filename)
            student_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])


    experience = defaultdict(list)
    start_epoch = 0 if checkpoint is None else checkpoint['step']
    for epoch in itertools.count(start_epoch):
        # Checkpointing
        if checkpoint_filename is not None and epoch % args.checkpoint_interval == 0 and epoch > start_epoch:
            torch.save({
                'model': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': epoch,
            }, checkpoint_filename)

        # Create new batch of experience every few epochs
        if epoch % args.epochs_per_dataset == 0:
            for env in envs:
                if not env.train_enabled:
                    continue
                if env.name is None:
                    raise ValueError('Environment must have a name')
                experience[env.name] = generate_experience(
                    teacher_models[env.name],
                    env.env,
                    num_episodes=args.num_episodes
                )

        # Evaluate the model on the environment
        filename = os.path.join(tempdir, f'results-{epoch}.pt')
        evaluate(
            model=student_model,
            env_config=args.env_config,
            model_config=args.student_model_config,
            num_episodes=10,
            results_filename=filename,
            tempdir=tempdir,
        )

        # Save results
        if wandb.run is not None:
            results = torch.load(filename)
            for env_key in results.keys():
                test_rewards = [r['episode_reward'].item() for r in results[env_key]]
                wandb.log({f'test_reward/{env_key}': np.mean(test_rewards)}, commit=False)

        # Train the model
        #for env in envs:
        #    train_single_model(
        #        experience[env.name],
        #        student_model[env.model_name] if env.model_name is not None else student_model,
        #        optimizer,
        #        batch_size=args.batch_size
        #    )
        train_multi_model(
            [experience[env.name] for env in envs if env.train_enabled],
            [student_model[env.model_name] if env.model_name is not None else student_model for env in envs if env.train_enabled],
            optimizer,
            batch_size=args.batch_size
        )

    if wandb.run is not None:
        wandb.log({}, commit=True)


if __name__ == '__main__':
    parser = init_arg_parser()
    args = parser.parse_args()
    main(args)


"""
TODO:
- Implement checkpointing [DONE]
- Implement training on multiple envs and submodels
"""
