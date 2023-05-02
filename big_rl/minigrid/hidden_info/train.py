import argparse
import itertools
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from big_rl.minigrid.hidden_info.hidden_info import OBJECTS


TARGET_KEYS = ['target_idx', 'target_pos', 'all_objects_pos', 'wall_map', 'object_presence']


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        for in_size, out_size in zip([input_size] + hidden_sizes, hidden_sizes + [output_size]):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(
            *layers[:-1]
        )

    def forward(self, x):
        return self.seq(x)


class Model2(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_sizes: dict):
        super().__init__()
        layers = []
        for in_size, out_size in zip([input_size] + hidden_sizes, hidden_sizes):
            layers.append(torch.nn.Linear(in_size, out_size))
            layers.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(
            *layers
        )
        last_layer_size = hidden_sizes[-1] if len(hidden_sizes) > 0 else input_size
        self.output_layers = torch.nn.ModuleDict({
            k: torch.nn.Linear(last_layer_size, v)
            for k, v in output_sizes.items()
        })

    def forward(self, x):
        x = self.seq(x)
        return {k: layer(x) for k, layer in self.output_layers.items()}


def make_loss(target_key):
    if target_key == 'target_idx':
        def loss(output, target):
            return torch.nn.functional.cross_entropy(output, target.squeeze(1))
        return loss
    elif target_key in ['all_objects_pos']:
        def loss(output, target):
            i = target != -1 # -1 means the object is not present, so they should not affect the loss
            l = torch.nn.functional.mse_loss(output, target.float(), reduction='none')
            return l[i].mean()
        return loss
    elif target_key in ['target_pos']:
        def loss(output, target):
            return torch.nn.functional.mse_loss(output, target.float())
        return loss
    elif target_key in ['wall_map', 'object_presence']:
        def loss(output, target):
            return torch.nn.functional.binary_cross_entropy_with_logits(output, target.float())
        return loss
    else:
        raise ValueError(f'Unknown target key: {target_key}')


def make_model(target_key, input_size):
    output_sizes = {
        'target_idx': len(OBJECTS),
        'target_pos': 2,
        'all_objects_pos': 2 * len(OBJECTS),
        'wall_map': 25*25,
        'object_presence': len(OBJECTS),
    }
    hidden_sizes = {
        'target_idx': [64],
        'target_pos': [64],
        'all_objects_pos': [64],
        'wall_map': [2048,2048],
        'object_presence': [64],
    }
    if target_key not in output_sizes:
        raise ValueError(f'Unknown target key: {target_key}')
    return Model(
            input_size=input_size,
            hidden_sizes=hidden_sizes[target_key],
            output_size=output_sizes[target_key]
    )


def main(train_dataset_filename, val_dataset_filename, target_key, model_filename, batch_size, num_epochs, lr, test_frequency, cuda):
    if cuda and torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    train_loss_history = []
    val_loss_history = []
    model = None
    loss_fn = make_loss(target_key)
    try:
        with h5py.File(train_dataset_filename, 'r') as dataset, \
                h5py.File(val_dataset_filename, 'r') as val_dataset:
            model = make_model(
                target_key=target_key,
                input_size=dataset['hidden'].shape[1], # type: ignore
            )
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            if num_epochs is None or num_epochs < 0:
                epoch_iterator = itertools.count()
            else:
                epoch_iterator = range(num_epochs)
            for epoch in tqdm(epoch_iterator):
                if epoch % test_frequency == 0:
                    with torch.no_grad():
                        hidden = torch.tensor(val_dataset['hidden'], dtype=torch.float, device=device) # type: ignore
                        target_output = torch.tensor(val_dataset[target_key], dtype=torch.long, device=device) # type: ignore

                        output = model(hidden)
                        loss = loss_fn(output, target_output)
                        val_loss_history.append((epoch, loss.item()))
                    if wandb.run is not None:
                        wandb.log({'validation loss': loss.item()})
                    tqdm.write(f'Validation loss: {loss.item()}')

                indices = sorted(np.random.choice(len(dataset['hidden']), size=batch_size, replace=False)) # type: ignore
                hidden = torch.tensor(dataset['hidden'][indices], dtype=torch.float, device=device) # type: ignore
                target_output = torch.tensor(dataset[target_key][indices], dtype=torch.long, device=device) # type: ignore

                optimizer.zero_grad()
                output = model(hidden)
                loss = loss_fn(output, target_output)
                loss.backward()
                optimizer.step()

                tqdm.write(f'Epoch {epoch}\t loss: {loss.item():.2f}')
                train_loss_history.append(loss.item())
                if wandb.run is not None:
                    wandb.log({'loss': loss.item()})
    except KeyboardInterrupt:
        pass

    if model is not None:
        torch.save(model.state_dict(), model_filename)
        print(f'Saved model to {os.path.abspath(model_filename)}')

    plt.plot(train_loss_history, label='train')
    plt.plot([x[0] for x in val_loss_history], [x[1] for x in val_loss_history], label='validation')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('loss.png')
    plt.close()


if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./dataset.h5',
                        help='Path to a file to load dataset.')
    parser.add_argument('--val-dataset', type=str, default='./val-dataset.h5',
                        help='Path to a file to load dataset.')
    parser.add_argument('--target-key', type=str, default='target_idx',
                        choices=TARGET_KEYS,
                        help='Key of the target to train on.')
    parser.add_argument('--model', type=str, default='./model.pt',
                        help='Path to a file to save the model.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--test-frequency', type=int, default=10,
                        help='Number of epochs between validation tests.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA.')
    parser.add_argument('--wandb', action='store_true',
                        help='Save results to W&B.')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='hidden_info')

    main(
            train_dataset_filename=args.dataset,
            val_dataset_filename=args.val_dataset,
            target_key=args.target_key,
            model_filename=args.model,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            test_frequency=args.test_frequency,
            lr=args.lr,
            cuda=args.cuda,
    )
