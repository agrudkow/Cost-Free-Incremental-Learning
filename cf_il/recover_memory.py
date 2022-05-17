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
        num_images_per_class: int = 10,
        scale: float = 1,  # TODO: try scale =0.1 ?
        eta: float = 0.7) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    criterion = nn.CrossEntropyLoss()
    fc_weights = model.fc.weight.detach().cpu().numpy()  # type: ignore[union-attr, operator]
    fc_weights = np.transpose(fc_weights)  # nasz pomysł
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
        for i in range(num_images_per_class):
            rand_img = np.random.uniform(0, 1, size=image_shape)
            # rand_img = np.random.normal(0.5, 0.5, size=image_shape)
            rand_img = rand_img.astype(np.float32)
            rand_img_tensor = torch.tensor(
                rand_img.reshape((1, *reversed(image_shape))),
                requires_grad=True,
                device=device,
            )

            # opt = torch.optim.Adam(lr=0.01, params=[rand_img_tensor])
            for _ in range(1500):
                # opt.zero_grad()
                logits = model(rand_img_tensor)

                loss = criterion(logits / 20, torch.tensor(pseudo_labels[i], device=device).reshape(1, -1))
                loss.backward()
                # opt.step()
                with torch.no_grad():
                    rand_img_tensor += rand_img_tensor.grad * 0.01
                    tensor_min = rand_img_tensor.min()
                    tensor_max = rand_img_tensor.max()
                    rand_img_tensor = (rand_img_tensor - tensor_min) / (tensor_max - tensor_min)
                rand_img_tensor.requires_grad = True
                # TODO: add assert on weights

            synth_logits = model(rand_img_tensor).detach().cpu()
            synth_image = rand_img_tensor.detach().cpu()

            # NOTE: display image
            # transform = SequentialCIFAR10.get_normalization_transform()
            # img = transform(synth_image[0])
            img = synth_image[0]
            img = np.transpose(img.numpy(), (1, 2, 0))
            # img_min = np.min(img, axis=(0, 1))
            # img_max = np.max(img, axis=(0, 1))
            # img = (img - img_min) / (img_max - img_min)
            plt.imshow(img)

            images_all.append(synth_image)
            logits_all.append(synth_logits)

    return (np.array(images_all, dtype=torch.TensorType), np.array(logits_all, dtype=torch.TensorType))
