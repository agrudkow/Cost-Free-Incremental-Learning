from argparse import Namespace
import os

import hydra
import torch
import torch.nn as nn

from cf_il.model import CFIL, convNet
from cf_il.train import train
from cf_il.conf.constants import ROOT_DIR
from cf_il.conf.config import Config
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from torchsummary import summary


@hydra.main(config_path=os.path.join(ROOT_DIR, "conf"), config_name="config")
def main(config: Config) -> CFIL:
    args: Namespace = config.cf_il  # type: ignore
    print(args)
    dataset = get_dataset(args)
    assert isinstance(dataset, ContinualDataset) is True
    assert dataset.N_TASKS is not None
    assert dataset.N_CLASSES_PER_TASK is not None
    num_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    image_shape = None
    if dataset.NAME == 'seq-cifar10':
        image_shape = (32, 32, 3)
    elif dataset.NAME =='seq-tinyimg':  # problem with loading this dataset
        image_shape = (64, 64, 3)
    else:
        raise ValueError('Image shape cannot be None.')

  
    if args.backbone=="convNet":
      net= convNet(10).cuda()
      dim1=4096
    elif args.backbone=="AlexNet":  ## not suitable for CIFAR
      net = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
      #import torchvision.models as models
      #net =models.alexnet()
    elif args.backbone=="ResNet18":
      net = torch.nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
      dim1=512
    elif args.backbone=="ResNet20":
      from pytorchcv.model_provider import get_model as ptcv_get_model
      net = ptcv_get_model("resnet20_cifar10", pretrained=True)
 
     

    # Load model
    backbone: backbone=net
    backbone.eval()
    if args.backbone!="ResNet20":
      backbone.fc = nn.Linear(dim1, num_classes)  

    summary(net.cuda(), image_shape)

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
        args=args,
    )

    return model


if __name__ == '__main__':
    main()
