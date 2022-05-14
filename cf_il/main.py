from argparse import ArgumentParser, Namespace

import sys    ###PT
sys.path.append("./")  ##PT #print(sys.path)


import torch
import torch.nn as nn
from cf_il.model import CFIL
from cf_il.train import train

from datasets import NAMES as DATASET_NAMES, get_dataset
from datasets.utils.continual_dataset import ContinualDataset

import wandb  ####### PT



parser = ArgumentParser(description='cf-lr', allow_abbrev=False)

# Logging
parser.add_argument(
    '--tensorboard',
    action='store_true',
    help='Enable tensorboard logging',
)

parser.add_argument(   ####### PT
    '--wandb',
    action='store_true',
    help='Enable logging in weights&biases',
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
    help='Degree of distillation.',
)


def main():
    args = parser.parse_known_args()[0]

    # args = fake_args(args)
    
    
    
    # Launch 5 simulated experiments
    #total_runs = 1
    #for run in range(total_runs):
      # Start a new run to track this script
      #wandb.init(
          # Set the project where this run will be logged
          #project="CF_IL", 
          # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
          #name=f"experiment_{run}", 
          # Track hyperparameters and run metadata
          #config={
          #"learning_rate": args.lr,
          #"scale": 1,
          #"dataset": "seq-cifar10"
          #})
  

    dataset = get_dataset(args)
    assert isinstance(dataset, ContinualDataset) is True
    assert dataset.N_TASKS is not None
    assert dataset.N_CLASSES_PER_TASK is not None
    num_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    image_shape = None
    if dataset.NAME == 'seq-cifar10':
        image_shape = (32, 32, 3)
    else:
        raise ValueError('Image shape cannot be None.')

    # Load model
    backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    backbone.eval()
    backbone.fc = nn.Linear(512, num_classes)

    model = CFIL(
        backbone=backbone,
        args=args,
        loss=nn.CrossEntropyLoss(),
        transform=dataset.get_transform(),
        image_shape=image_shape,
    )

    train(
        model=model,
        dataset=dataset,
        args=args,# wandb_writer=wandb,
    )


if __name__ == '__main__':
    main()
