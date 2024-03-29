"""
`./configs/envs/train/1/` contains configurations for individual tasks.
This script takes these configurations and generates configurations for groups of tasks.
Pairs of tasks go in `./configs/envs/train/2/`, triples of tasks go in `./configs/envs/train/3/`, etc.
"""

import copy
import os
import yaml
import itertools


ENV_CONFIG_DIR = './big_rl_experiments/exp1/configs/envs'


def get_single_task_configs():
    directory = os.path.join(ENV_CONFIG_DIR, 'train/1')
    filenames = os.listdir(directory)
    filenames = [f for f in filenames if f.endswith('.yaml')]
    return filenames


def main():
    filenames = sorted(get_single_task_configs())

    # Load all the configs
    configs = {}
    for filename in filenames:
        with open(os.path.join(ENV_CONFIG_DIR, 'train/1', filename), 'r') as f:
            configs[filename] = yaml.safe_load(f)

    ## Generate training configs (merge into one large AsyncVectorEnv)
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'train/autogenerated')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        merged_config = copy.deepcopy(configs_subset[0])
    #        for c in configs_subset[1:]:
    #            merged_config[0]['envs'].extend(c[0]['envs'])
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(merged_config, f)

    ## Generate training configs (merge, but keep each task as a distinct AsyncVectorEnv)
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'train/autogenerated')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        merged_config = []
    #        for c in configs_subset:
    #            merged_config.extend(c)
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(merged_config, f)

    ## Generate evaluation configs
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'test/autogenerated')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        separated_configs = []
    #        for config in configs_subset:
    #            assert len(config) == 1
    #            for env in config[0]['envs']:
    #                separated_configs.extend(copy.deepcopy(config))
    #                separated_configs[-1]['name'] = env['name']
    #                separated_configs[-1]['type'] = 'SyncVectorEnv'
    #                env = copy.deepcopy(env)
    #                env['repeat'] = 1
    #                env['kwargs']['render_mode'] = 'rgb_array'
    #                separated_configs[-1]['envs'] = [env]
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(separated_configs, f)

    ## Generate evaluation configs with randomized observations
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'test/autogenerated/shuffled_obs')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        separated_configs = []
    #        for config in configs_subset:
    #            assert len(config) == 1
    #            for env in config[0]['envs']:
    #                separated_configs.extend(copy.deepcopy(config))
    #                separated_configs[-1]['name'] = env['name']
    #                separated_configs[-1]['type'] = 'SyncVectorEnv'
    #                env = copy.deepcopy(env)
    #                env['repeat'] = 1
    #                env['kwargs']['render_mode'] = 'rgb_array'
    #                env['wrappers'].insert(1, {'type': 'ShuffleObservation'}) # <-- Most important line
    #                separated_configs[-1]['envs'] = [env]
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(separated_configs, f)

    ## Generate evaluation configs with randomized action
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'test/autogenerated/shuffled_action')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        separated_configs = []
    #        for config in configs_subset:
    #            assert len(config) == 1
    #            for env in config[0]['envs']:
    #                separated_configs.extend(copy.deepcopy(config))
    #                separated_configs[-1]['name'] = env['name']
    #                separated_configs[-1]['type'] = 'SyncVectorEnv'
    #                env = copy.deepcopy(env)
    #                env['repeat'] = 1
    #                env['kwargs']['render_mode'] = 'rgb_array'
    #                env['wrappers'].insert(1, {'type': 'ShuffleAction'}) # <-- Most important line
    #                separated_configs[-1]['envs'] = [env]
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(separated_configs, f)

    ## Generate evaluation configs with randomized observations and actions
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'test/autogenerated/shuffled_obs_and_action')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        separated_configs = []
    #        for config in configs_subset:
    #            assert len(config) == 1
    #            for env in config[0]['envs']:
    #                separated_configs.extend(copy.deepcopy(config))
    #                separated_configs[-1]['name'] = env['name']
    #                separated_configs[-1]['type'] = 'SyncVectorEnv'
    #                env = copy.deepcopy(env)
    #                env['repeat'] = 1
    #                env['kwargs']['render_mode'] = 'rgb_array'
    #                env['wrappers'].insert(1, {'type': 'ShuffleObservation'}) # <-- Most important line
    #                env['wrappers'].insert(1, {'type': 'ShuffleAction'}) # <-- Most important line
    #                separated_configs[-1]['envs'] = [env]
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(separated_configs, f)

    ## Generate evaluation configs with observations completely occluded
    #for num_tasks in range(1, len(filenames) + 1):
    #    directory = os.path.join(ENV_CONFIG_DIR, f'test/autogenerated/occluded_obs_100')
    #    os.makedirs(directory, exist_ok=True)
    #    for subset in itertools.combinations(filenames, num_tasks):
    #        filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
    #        filename = os.path.join(directory, filename)
    #        configs_subset = [configs[n] for n in subset]
    #        separated_configs = []
    #        for config in configs_subset:
    #            assert len(config) == 1
    #            for env in config[0]['envs']:
    #                separated_configs.extend(copy.deepcopy(config))
    #                separated_configs[-1]['name'] = env['name']
    #                separated_configs[-1]['type'] = 'SyncVectorEnv'
    #                env = copy.deepcopy(env)
    #                env['repeat'] = 1
    #                env['kwargs']['render_mode'] = 'rgb_array'
    #                env['wrappers'].insert(1, {
    #                    'type': 'OccludeObservation',
    #                    'kwargs': {
    #                        'p': 1.0,
    #                        'value': 0.0,
    #                    }
    #                }) # <-- Most important line
    #                separated_configs[-1]['envs'] = [env]
    #        print(f'Writing {filename}')
    #        with open(filename, 'w') as f:
    #            yaml.dump(separated_configs, f)

    # Generate evaluation configs with observations completely occluded, including the action and reward
    for num_tasks in range(1, len(filenames) + 1):
        directory = os.path.join(ENV_CONFIG_DIR, f'test/autogenerated/occluded_obs_action_reward_100')
        os.makedirs(directory, exist_ok=True)
        for subset in itertools.combinations(filenames, num_tasks):
            filename = ''.join(['1' if n in subset else '0' for n in filenames]) + '.yaml'
            filename = os.path.join(directory, filename)
            configs_subset = [configs[n] for n in subset]
            separated_configs = []
            for config in configs_subset:
                assert len(config) == 1
                for env in config[0]['envs']:
                    separated_configs.extend(copy.deepcopy(config))
                    separated_configs[-1]['name'] = env['name']
                    separated_configs[-1]['type'] = 'SyncVectorEnv'
                    env = copy.deepcopy(env)
                    env['repeat'] = 1
                    env['kwargs']['render_mode'] = 'rgb_array'
                    env['wrappers'].insert(5, {
                        'type': 'OccludeObservation',
                        'kwargs': {
                            'p': 1.0,
                            'value': 0.0,
                        }
                    }) # <-- Most important line
                    separated_configs[-1]['envs'] = [env]
            print(f'Writing {filename}')
            with open(filename, 'w') as f:
                yaml.dump(separated_configs, f)

    print('Done')


if __name__ == '__main__':
    main()
