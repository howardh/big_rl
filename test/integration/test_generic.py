import pytest
import textwrap

from big_rl.generic.script import init_arg_parser, main

##################################################
# Test all combinations of observation and action spaces

"""
                    Action space
          /Discrete
          |   /Box
          |   |   /Dict
Obs Space |   |   |
----------|---|---|-----------------------------
Discrete  |   |   |
Box       |   | X |
Dict      | X | X |
Tuple     |   |   |

"""

@pytest.mark.timeout(30)
def test_dict_obs_box_action_recurrent(tmpdir):
    model_config = textwrap.dedent("""
        type: ModularModel1
        key_size: 32
        value_size: 32
        num_heads: 2
        input_modules:
          obs:
            type: LinearInput
            kwargs:
              input_size:
                source: 'observation_space'
                accessor: '["obs"].high.shape[0]'
          reward:
            type: ScalarInput
          done:
            type: IgnoredInput
          action:
            type: LinearInput
            kwargs:
              input_size:
                source: 'action_space'
                accessor: '.high.shape[0]'
        output_modules:
          value:
            type: LinearOutput
            kwargs:
              output_size: 1
          action_mean:
            type: LinearOutput
            kwargs:
              output_size:
                source: 'action_space'
                accessor: '.high.shape[0]'
          action_logstd:
            type: LinearOutput
            kwargs:
              output_size:
                source: 'action_space'
                accessor: '.high.shape[0]'
        core_modules:
          type: RecurrentAttention17
          kwargs:
            ff_size: []
            num_modules: 1
    """.strip('\n'))
    model_config_file = tmpdir.join('model_config.yaml')
    model_config_file.write(model_config)

    env_config = textwrap.dedent("""
        - type: SyncVectorEnv # AsyncVectorEnv, SyncVectorEnv, or Env
          envs:
          - &env
            name: Hopper
            kwargs: &kwargs # Arguments for gymnasium.make
              id: Hopper-v4
            wrappers:
            - type: MetaWrapper
              kwargs: 
                episode_stack: 1
                dict_obs: true
                randomize: false
    """.strip('\n'))
    env_config_file = tmpdir.join('env_config.yaml')
    env_config_file.write(env_config)

    # Run test
    parser = init_arg_parser()
    args = parser.parse_args(f'--env-config {env_config_file} --model-config {model_config_file} --max-steps 100 --rollout-length 10'.split())
    main(args)


@pytest.mark.timeout(30)
def test_box_obs_box_action_recurrent(tmpdir):
    model_config = textwrap.dedent("""
        type: LSTMModel2
        kwargs:
          input_size: 
            source: 'observation_space'
            accessor: '.shape[0]'
          action_size:
            source: 'action_space'
            accessor: '.shape[0]'
          ff_size: [64, 64]
    """.strip('\n'))
    model_config_file = tmpdir.join('model_config.yaml')
    model_config_file.write(model_config)

    env_config = textwrap.dedent("""
        - type: SyncVectorEnv # AsyncVectorEnv, SyncVectorEnv, or Env
          envs:
          - &env
            name: Hopper
            kwargs: &kwargs # Arguments for gymnasium.make
              id: Hopper-v4
    """.strip('\n'))
    env_config_file = tmpdir.join('env_config.yaml')
    env_config_file.write(env_config)

    parser = init_arg_parser()
    args = parser.parse_args(f'--env-config {env_config_file} --model-config {model_config_file} --max-steps 100 --rollout-length 10'.split())
    main(args)


@pytest.mark.timeout(30)
@pytest.mark.skip(reason="Need to implement model first")
def test_box_obs_discrete_action_recurrent(tmpdir):
    model_config = textwrap.dedent("""
        ...
    """.strip('\n'))
    model_config_file = tmpdir.join('model_config.yaml')
    model_config_file.write(model_config)

    env_config = textwrap.dedent("""
        - type: SyncVectorEnv # AsyncVectorEnv, SyncVectorEnv, or Env
          envs:
          - kwargs: &kwargs # Arguments for gymnasium.make
              id: ALE/Pong-v5
              full_action_space: true
              frameskip: 1
              difficulty: 1
            repeat: 4 # Number of copies of this environment in the VectorEnv
            wrappers:
            - type: AtariPreprocessing
              kwargs:
                screen_size: 84
            - type: FrameStack
              kwargs:
                num_stack: 1
    """.strip('\n'))
    env_config_file = tmpdir.join('env_config.yaml')
    env_config_file.write(env_config)

    parser = init_arg_parser()
    args = parser.parse_args(f'--env-config {env_config_file} --model-config {model_config_file} --max-steps 100 --rollout-length 10'.split())
    main(args)


@pytest.mark.timeout(30)
def test_dict_obs_discrete_action_recurrent(tmpdir):
    model_config = textwrap.dedent("""
        type: ModularModel1
        input_modules:
          'obs':
            type: ImageInput84
            kwargs:
              in_channels: 1
          reward:
            type: ScalarInput
          done:
            type: IgnoredInput
          action:
            type: DiscreteInput
            kwargs:
              input_size:
                source: 'action_space'
                accessor: '.n'
        output_modules:
          value:
            type: LinearOutput
            kwargs:
              output_size: 1
          action:
            type: LinearOutput
            kwargs:
              output_size:
                source: 'action_space'
                accessor: '.n'
        core_modules:
          type: RecurrentAttention17
          kwargs:
            ff_size: []
            num_modules: 1
    """.strip('\n'))
    model_config_file = tmpdir.join('model_config.yaml')
    model_config_file.write(model_config)

    env_config = textwrap.dedent("""
        - type: SyncVectorEnv # AsyncVectorEnv, SyncVectorEnv, or Env
          envs:
          - &env
            name: Boxing
            kwargs: &kwargs # Arguments for gymnasium.make
              id: ALE/Boxing-v5
              full_action_space: true
              frameskip: 1
              difficulty: 1
            repeat: 4 # Number of copies of this environment in the VectorEnv
            wrappers:
            - type: AtariPreprocessing
              kwargs:
                screen_size: 84
            - type: FrameStack
              kwargs:
                num_stack: 1
            - type: MetaWrapper
              kwargs: 
                episode_stack: 1
                dict_obs: true
                randomize: false
    """.strip('\n'))
    env_config_file = tmpdir.join('env_config.yaml')
    env_config_file.write(env_config)

    parser = init_arg_parser()
    args = parser.parse_args(f'--env-config {env_config_file} --model-config {model_config_file} --max-steps 100 --rollout-length 10'.split())
    main(args)
