from argparse import ArgumentParser


def init_parser_trainer(parser: ArgumentParser):
    parser.add_argument('--max-steps', type=int, default=0, help='Number of transitions to train for. If 0, train forever.')
    parser.add_argument('--max-steps-total', type=int, default=0, help='Number of transitions to train for. Unlike `--max-steps`, this takes into account all steps run on previous executions, so if an experiment was interrupted and resumed, then steps run before the interruption will also be counted towards this limit.')

    parser.add_argument('--optimizer', type=str, default='RMSprop', help='Optimizer', choices=['Adam', 'RMSprop', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')

    parser.add_argument('--rollout-length', type=int, default=128, help='Length of rollout.')
    parser.add_argument('--reward-clip', type=float, default=1, help='Clip the reward magnitude to this value.')
    parser.add_argument('--reward-scale', type=float, default=1, help='Scale the reward magnitude by this value.')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for GAE.')
    parser.add_argument('--norm-adv', type=bool, default=True, help='Normalize the advantages.')
    parser.add_argument('--clip-vf-loss', type=float, default=0.1, help='Clip the value function loss.')
    parser.add_argument('--vf-loss-coeff', type=float, default=0.5, help='Coefficient for the value function loss.')
    parser.add_argument('--entropy-loss-coeff', type=float, default=0.01, help='Coefficient for the entropy loss.')
    parser.add_argument('--target-kl', type=float, default=None, help='Target KL divergence.')
    parser.add_argument('--num-epochs', type=int, default=5, help='Number of minibatches.')
    parser.add_argument('--minibatch-size', type=int, default=256, help='Minibatch size. Only applies to the non-recurrent baseline model.')
    parser.add_argument('--num-minibatches', type=int, default=4, help='Number of minibatches. Only applies to the non-recurrent baseline model.')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm.')
    parser.add_argument('--warmup-steps', type=int, default=0, help='Number of warmup steps on the environment before we start training on the generated samples.')
    parser.add_argument('--update-hidden-after-grad', action='store_true', help='Update the hidden state after the gradient step.')
    parser.add_argument('--backtrack', action='store_true', help='Backtrack if the KL constraint is violated.')

    parser.add_argument('--random-score', type=float, nargs='*', default=None, help='Random score for each environment.')
    parser.add_argument('--max-score', type=float, nargs='*', default=None, help='Max score for each environment.')
    parser.add_argument('--multitask-dynamic-weight', action='store_true', help='Use dynamic weight for multitask learning.')
    parser.add_argument('--multitask-dynamic-weight-temperature', type=float, default=10., help='Temperature for dynamic weight for multitask learning.')
    parser.add_argument('--multitask-static-weight', type=float, nargs='*', default=None, help='Relative weight of each task. If the sum is not 1, they will be normalized.')

    parser.add_argument('--l2-reg', type=float, default=0, help='L2 regularization weight. Default is 0.')


def init_parser_model(parser: ArgumentParser):
    parser.add_argument('--model-type', type=str, default='ModularPolicy5',
                        help='Model type', choices=['ModularPolicy5', 'ModularPolicy5LSTM', 'ModularPolicy7', 'ModularPolicy8', 'Baseline'])
    parser.add_argument('--recurrence-type', type=str,
                        default='RecurrentAttention14',
                        help='Recurrence type',
                        choices=[f'RecurrentAttention{i}' for i in [11,14,15,16]])
    parser.add_argument('--architecture', type=int,
                        default=[3,3], nargs='*',
                        help='Size of each layer in the model\'s core')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Size of the model\'s hidden state. Only applies to LSTM models.')
    parser.add_argument('--ff-size', type=int, nargs='*', default=[1024],
                        help='Size of the model\'s fully connected feedforward layers. Only applies to attention models.')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to a model config file (yaml format). If specified, all other model arguments are ignored.')
