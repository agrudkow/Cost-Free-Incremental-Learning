from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from cf_il.model import CFIL
from cf_il.train import train

from datasets import NAMES as DATASET_NAMES, get_dataset
from datasets.utils.continual_dataset import ContinualDataset

parser = ArgumentParser(description='cf-lr', allow_abbrev=False)

# Logging
parser.add_argument(
    '--tensorboard',
    action='store_true',
    help='Enable tensorboard logging',
)

# Dataset
parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    choices=DATASET_NAMES,
    help='Which dataset to perform experiments on.',
)
parser.add_argument(
    '--validation',
    action='store_true',
    help='Test on the validation set',
)

# Experiments setup
parser.add_argument(
    '--lr',
    type=float,
    required=True,
    help='Learning rate.',
)
parser.add_argument(
    '--momentum',
    type=float,
    required=True,
    help='Momentum.',
)
parser.add_argument(
    '--batch_size',
    type=int,
    required=True,
    help='Batch size.',
)
parser.add_argument(
    '--n_epochs',
    type=int,
    required=True,
    help='The number of epochs for each task.',
)

# Memory buffer
parser.add_argument(
    '--buffer_size',
    type=int,
    required=True,
    help='The size of the memory buffer.',
)
parser.add_argument(
    '--minibatch_size',
    type=int,
    required=True,
    help='The batch size of the memory buffer.',
)

# CF-IL
parser.add_argument(
    '--alpha',
    type=float,
    required=True,
    help='Penalty weight.',
)


def fake_args(args: Namespace) -> Namespace:
    args.tensorboard = args.tensorboard or True
    args.dataset = args.dataset or 'seq-cifar10'
    args.lr = args.lr or 0.001
    args.momentum = args.momentum or 0.9
    args.n_epochs = args.n_epochs or 1
    args.batch_size = args.batch_size or 10
    args.minibatch_size = args.minibatch_size or 10
    args.buffer_size = args.buffer_size or 100
    args.alpha = args.alpha or 0.2

    return args


def main():
    args = parser.parse_known_args()[0]

    # args = fake_args(args)

    dataset = get_dataset(args)
    assert isinstance(dataset, ContinualDataset) is True
    assert dataset.N_TASKS is not None
    assert dataset.N_CLASSES_PER_TASK is not None
    num_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    # Load model
    backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # False
    backbone.eval()
    backbone.fc = nn.Linear(512, num_classes)

    model = CFIL(
        backbone=backbone,
        args=args,
        loss=nn.CrossEntropyLoss(),
        transform=dataset.get_transform(),
    )

    train(
        model=model,
        dataset=dataset,
        args=args,
    )


if __name__ == '__main__':
    main()
