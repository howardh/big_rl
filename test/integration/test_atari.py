import pytest
import textwrap

from big_rl.atari.script import init_arg_parser, main


@pytest.fixture
def model_config_file(tmpdir):
    config = textwrap.dedent("""
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
    config_file = tmpdir.join('model_config.yaml')
    config_file.write(config)
    return config_file


@pytest.fixture
def env_config_file(tmpdir):
    config = textwrap.dedent("""
        - type: AsyncVectorEnv # AsyncVectorEnv, SyncVectorEnv, or Env
          envs:
          - &env
            name: Boxing-m0d1
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
    config_file = tmpdir.join('env_config.yaml')
    config_file.write(config)
    return config_file


@pytest.mark.timeout(30)
def test_atari_script():
    parser = init_arg_parser()
    args = parser.parse_args('--envs ALE/VideoPinball-v5 --num-envs 2 --model-type ModularPolicy8 --recurrence-type RecurrentAttention16 --architecture 1 --ff-size --max-steps 100 --rollout-length 10'.split())
    main(args)


@pytest.mark.timeout(30)
def test_config_file(model_config_file, env_config_file):
    parser = init_arg_parser()
    args = parser.parse_args(f'--env-config {env_config_file} --model-config {model_config_file} --max-steps 100 --rollout-length 10'.split())
    main(args)
