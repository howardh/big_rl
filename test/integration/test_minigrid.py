import pytest

from big_rl.minigrid.script import init_arg_parser, main

@pytest.mark.timeout(30)
def test_config_file():
    parser = init_arg_parser()
    args = parser.parse_args(f'--envs fetch-004-stop_100_trials	--num-envs 3 --model-type ModularPolicy5LSTM --hidden-size 8  --max-steps 100 --rollout-length 10'.split())
    main(args)

