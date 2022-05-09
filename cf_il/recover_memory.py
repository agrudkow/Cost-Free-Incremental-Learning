from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing as pp
import torch
import torch.nn as nn

from cf_il.generate_dirichlet import generate_dirichlet


def recovery_memory(
        model: nn.Module,
        num_tasks: int,
        num_classes_per_task: int = 2,
        num_images_per_class: int = 10,
        scale: float = 1,  # TODO: try scale =0.1 ?
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    criterion = nn.CrossEntropyLoss()
    fc_weights = model.fc.weight.detach().numpy()  # type: ignore[union-attr, operator]
    fc_weights = np.transpose(fc_weights)  # nasz pomysł
    fc_weight_norm = pp.normalize(fc_weights, axis=0)

    sim_matrix = np.matmul(np.transpose(fc_weight_norm), fc_weight_norm)

    norm_similarity_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))

    images_all: List[npt.NDArray[np.float32]] = []
    logits_all: List[npt.NDArray[np.float32]] = []

    for classes in range(num_tasks * num_classes_per_task):
        pseudo_labels = generate_dirichlet(
            batch_size=num_images_per_class,
            class_id=classes,
            scale=scale,
            similarity_matrix=norm_similarity_matrix,
        )
        for i in range(num_images_per_class):
            rand_img = np.random.normal(0.5, 0.5, size=(32, 32, 3))
            rand_img = rand_img.astype(np.float32)
            rand_img_tensor = torch.tensor(rand_img.reshape((1, 3, 32, 32)), requires_grad=True)

            logits = model(rand_img_tensor)

            loss = criterion(logits, torch.tensor(pseudo_labels[i]).reshape(1, -1))
            loss.backward()

            # TODO: add assert on weights

            synth_image = rand_img_tensor + rand_img_tensor.grad
            synth_logits = model(synth_image).detach().numpy()  # TODO: add loop for generating more images
            synth_image = synth_image.detach().numpy()

            images_all.append(synth_image)
            logits_all.append(synth_logits)

    return (np.array(images_all), np.array(logits_all))
