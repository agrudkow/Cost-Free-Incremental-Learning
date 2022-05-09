from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from cf_il.recover_memory import recovery_memory as rec_mem
from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer


class CFIL(ContinualModel):
    NAME: str = 'cf-il'  # type: ignore[assignment]
    COMPATIBILITY = [
        'class-il',
    ]

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        args: Namespace,
        transform: torchvision.transforms,
    ):
        super(CFIL, self).__init__(backbone, loss, args, transform)
        # Buffer contains synthetic images from RMP
        self.buffer = Buffer(
            buffer_size=self.args.buffer_size,
            device=self.device,
            mode='reservoir',
        )

        self.opt = optim.SGD(
            backbone.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
        )

    def observe(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> float:

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        print(f'Real dataset loss: {loss}')

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(self.args.minibatch_size)
            buf_outputs = self.net(buf_inputs)
            synth_loss = F.mse_loss(buf_outputs, buf_logits)
            print(f'Synthetic dataset loss: {synth_loss}')
            loss += self.args.alpha * synth_loss

        print(f'Total loss: {loss}')

        loss.backward()
        self.opt.step()

        return loss.item()  # type: ignore[no-any-return]

    def recover_memory(
        self,
        num_tasks: int,
        num_classes_per_task: int = 2,
        num_images_per_class: int = 10,
        scale: float = 1,
    ) -> None:
        self.buffer.empty()

        synth_images, synth_logits = rec_mem(
            model=self.net,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            num_images_per_class=num_images_per_class,
            scale=scale,
        )

        for img, logits in zip(synth_images, synth_logits):
            self.buffer.add_data(examples=img, logits=logits)
