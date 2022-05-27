import os
import random
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing as pp
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from cf_il.generate_dirichlet import generate_dirichlet
from datasets.seq_cifar10 import SequentialCIFAR10


def recovery_memory(
    model: nn.Module,
    num_classes: int,
    image_shape: Tuple[int, int, int],
    device: torch.device,
    scale: Tuple[float, float],
    buffer_size: int,
    eta: float,
    tau: float,
    optim_steps: int,
    optim_lr: float,
    synth_img_save_dir: Optional[str],
    synth_img_save_num: Optional[int],
    dirichlet_max_iter: int,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Recover memory for provided model.

    Args:
        model (nn.Module): Neural network as Torch Module.
        num_classes (int): Number of classes which has already been learn by the model.
        image_shape (Tuple[int, int, int]): Image shape of current dataset.
        device (torch.device): Device for torch tensors.
        scale (Tuple[float, float]): Tuple of scales to be used during computing center of the mass for the Dirichlet
            distribution. First element will be used for the first half of the classes and second for the rest.
        buffer_size (int): Size of the buffer for data impressions.
        eta (float): Parameter controlling how different sampled logit cant be from original class representation
            vector.
        tau (float): Cross Entropy temperature.
        optim_steps (int): Number of optimization step for generating data impressions.
        optim_lr (float): Learning rate for optimizer for generating data impressions
        synth_img_save_dir (Optional[str]): Dir for saving generated data impressions. If `None` data impressions
            are not saved.
        synth_img_save_num (Optional[int]): Number of saved data impressions. Data impressions are randomly selected.
        dirichlet_max_iter (int): Maximal amount of iterations of sampling for one logit.

    Returns:
        Tuple[npt.NDArray[Any], npt.NDArray[Any]]: Recovered memory in form of synthetic images and logits.
    """
    num_images_per_class = buffer_size // num_classes

    criterion = nn.CrossEntropyLoss()

    fc_weights = model.fc.weight.detach().cpu().numpy()  # type: ignore[union-attr, operator]
    fc_weights = np.transpose(fc_weights)
    fc_weight_norm = pp.normalize(fc_weights, axis=0)
    sim_matrix = np.matmul(np.transpose(fc_weight_norm), fc_weight_norm)

    images_all: List[torch.Tensor] = []
    logits_all: List[torch.Tensor] = []

    for classes in range(num_classes):
        pseudo_labels = generate_dirichlet(
            batch_size=num_images_per_class,
            class_id=classes,
            scale=scale,
            similarity_matrix=sim_matrix,
            eta=eta,
            max_iter=dirichlet_max_iter,
        )
        rand_img = np.random.uniform(0, 1, size=(num_images_per_class, *reversed(image_shape)))
        rand_img = rand_img.astype(np.float32)
        rand_img_tensor = torch.tensor(
            rand_img,
            requires_grad=True,
            device=device,
        )

        opt = torch.optim.Adam(lr=optim_lr, params=[rand_img_tensor])

        for i in range(optim_steps):
            if i % 100 == 0:
                print(f'Synth image optim step {i+1}/{optim_steps}')
            opt.zero_grad()
            logits = model(rand_img_tensor)

            loss = -criterion(logits / tau, torch.tensor(pseudo_labels, device=device))
            loss.backward()
            opt.step()

        synth_logits = model(rand_img_tensor).detach().cpu()
        synth_images = rand_img_tensor.detach().cpu()

        # NOTE: save image
        if synth_img_save_dir and synth_img_save_num:
            save_rand_images(
                images=synth_images,
                samples_num=synth_img_save_num,
                task=(num_classes // 2),
                cls_id=classes,
                dir=synth_img_save_dir,
            )

        for i, l in zip(synth_images, synth_logits):
            images_all.append(i.reshape((1, *reversed(image_shape))))  # type: ignore
            logits_all.append(l.reshape((1, sim_matrix.shape[0])))

    return (np.array(images_all, dtype=torch.TensorType), np.array(logits_all, dtype=torch.TensorType))


def save_rand_images(
    images: torch.Tensor,
    cls_id: int,
    task: int,
    samples_num: int,
    dir: str,
) -> None:
    """
    Save random data impressions.

    Args:
        images (torch.Tensor): Tensor with generated data impressions.
        cls_id (int): Class ID. Needed for file name.
        task (int): Task number. Needed for file name.
        samples_num (int): Number of data impressions to save.
        dir (str): Dir for saving data impressions.
    """
    transform_denorm = SequentialCIFAR10.get_denormalization_transform()

    images_ids = random.sample(range(len(images)), samples_num)
    for id in images_ids:
        img = transform_denorm(images[id])
        img = np.transpose(img.numpy(), (1, 2, 0))
        img_min = np.min(img, axis=(0, 1))
        img_max = np.max(img, axis=(0, 1))
        img = (img - img_min) / (img_max - img_min)

        save_time = datetime.now()
        save_time = datetime.strftime(save_time, "%Y_%m_%d_%H_%M_%S")  # type: ignore
        path = os.path.join(dir, f'synth_image_task-{task}_class-{cls_id}_{id}_{save_time}.png')

        plt.imsave(path, img)
