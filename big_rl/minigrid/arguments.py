from argparse import ArgumentParser


def init_parser_trainer(parser: ArgumentParser):
    parser.add_argument('--max-steps', type=int, default=0, help='Number of training steps to run. One step is one weight update.')

    parser.add_argument('--optimizer', type=str, default='RMSprop', help='Optimizer', choices=['Adam', 'RMSprop'])
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
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm.')
    parser.add_argument('--warmup-steps', type=int, default=0, help='Number of warmup steps on the environment before we start training on the generated samples.')


def init_parser_model(parser: ArgumentParser):
    parser.add_argument('--model-type', type=str, default='ModularPolicy5',
                        help='Model type', choices=['ModularPolicy5'])
    parser.add_argument('--recurrence-type', type=str,
                        default='RecurrentAttention14',
                        help='Recurrence type',
                        choices=[f'RecurrentAttention{i}' for i in [11,14]])
    parser.add_argument('--architecture', type=int,
                        default=[3,3], nargs='*',
                        help='Size of each layer in the model\'s core')


