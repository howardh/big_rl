
## Arguments
`--run` for rendering a simple
`--train` for training an example model using stable_baselines3
`--evaluate [MODEL LOCATION]` evaluate a model, must be combined with `--model_type [FILE TYPE]`

## Example of an Experiment with Mountain Car
`python big_rl/mountaincar/envs/__init__.py --train --evaluate "big_rl/model/dqn_mountaincar" --model_type ".zip" `