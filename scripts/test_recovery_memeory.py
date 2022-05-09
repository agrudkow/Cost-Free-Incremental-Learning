from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from cf_il.recover_memory import recovery_memory
from datasets import SequentialCIFAR10
from utils.args import add_management_args

# Add args parser
parser = ArgumentParser(description='cflr', allow_abbrev=False)

add_management_args(parser)
args = parser.parse_known_args()[0]

# fake args
setattr(args, "validation", True)
setattr(args, "batch_size", 32)

# Load dataset
cif = SequentialCIFAR10(args)

num_classes = cif.N_CLASSES_PER_TASK * cif.N_TASKS

# Load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # False
model.eval()
model.fc = nn.Linear(512, num_classes)

img_noise = recovery_memory(
    model=model,
    num_tasks=cif.N_TASKS,
    num_images_per_class=10,
)[0][0].reshape(32, 32, 3)

print(img_noise.shape)
#img_noise=img_noise.astype(np.float64)
plt.imshow(transforms.ToPILImage()(img_noise.astype('uint8')))  # unit8 jest dziwny bez niego wyżej działa
# https://stackoverflow.com/questions/62617533/trouble-with-pytorchs-topilimage
plt.show()