from typing import Any, List, Tuple
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing as pp
import torch
import torch.nn as nn

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
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    num_images_per_class = buffer_size // num_classes

    transform_denorm = SequentialCIFAR10.get_denormalization_transform()
    criterion = nn.CrossEntropyLoss()

    fc_weights = model.fc.weight.detach().cpu().numpy()  # type: ignore[union-attr, operator]
    fc_weights = np.transpose(fc_weights)
    fc_weight_norm = pp.normalize(fc_weights, axis=0)
    sim_matrix = np.matmul(np.transpose(fc_weight_norm), fc_weight_norm)
    norm_similarity_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))

    images_all: List[torch.Tensor] = []
    logits_all: List[torch.Tensor] = []

    for classes in range(num_classes):
        pseudo_labels = generate_dirichlet(
            batch_size=num_images_per_class,
            class_id=classes,
            scale=scale,
            similarity_matrix=norm_similarity_matrix,
            eta=eta,
        )
        rand_img = np.random.uniform(0, 1, size=(num_images_per_class, *reversed(image_shape)))
        rand_img = rand_img.astype(np.float32)
        rand_img_tensor = torch.tensor(
            rand_img,
            requires_grad=True,
            device=device,
        )

        opt = torch.optim.Adam(lr=0.01, params=[rand_img_tensor])
        for _ in range(1500):
            opt.zero_grad()
            logits = model(rand_img_tensor)

            loss = -criterion(logits / tau, torch.tensor(pseudo_labels, device=device))
            loss.backward()
            opt.step()
            
            with torch.no_grad(): ### just for verification of different variants
              img_min = np.min(rand_img_tensor.detach().cpu().numpy(), axis=(2,3))
              img_min=np.expand_dims(img_min, axis=(2,3))
              img_max = np.max(rand_img_tensor.detach().cpu().numpy(), axis=(2,3))
              img_max=np.expand_dims(img_max, axis=(2,3))
              r=rand_img_tensor.detach().cpu().numpy()
              rand_img_tensor = (r - img_min) / (img_max - img_min)
              rand_img_tensor=torch.tensor(
              rand_img_tensor,
              requires_grad=True,
              device=device,
              )

        synth_logits = model(rand_img_tensor).detach().cpu()
        synth_image = rand_img_tensor.detach().cpu()

        for i in range(num_images_per_class):
          img = transform_denorm(synth_image[i])
          img = np.transpose(img.numpy(), (1, 2, 0))
          plt.imshow(img)
          plt.savefig(f"/content/img_{classes}_{i}.jpg")

        # TODO: it is rather slow
        for i, l in zip(synth_image, synth_logits):
            images_all.append(i.reshape((1, *reversed(image_shape))))
            logits_all.append(l.reshape((1, sim_matrix.shape[0])))
        # images_all = images_all + [*synth_image]
        # logits_all = logits_all + [*synth_logits]

    return (np.array(images_all, dtype=torch.TensorType), np.array(logits_all, dtype=torch.TensorType))
