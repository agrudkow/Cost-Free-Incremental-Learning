from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing as pp
import torch
import torch.nn as nn

from cf_il.generate_dirichlet import generate_dirichlet


def recovery_memory(
    model: nn.Module,
    num_classes: int,
    image_shape: Tuple[int, int, int],
    device: torch.device,
    num_images_per_class: int = 10,
    scale: float = 1,  # TODO: try scale =0.1 ?
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    criterion = nn.CrossEntropyLoss()
    fc_weights = model.fc.weight.detach().cpu().numpy()  # type: ignore[union-attr, operator]
    fc_weights = np.transpose(fc_weights)  # nasz pomysł
    fc_weight_norm = pp.normalize(fc_weights, axis=0)

    sim_matrix = np.matmul(np.transpose(fc_weight_norm), fc_weight_norm)

    norm_similarity_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))

    images_all: List[npt.NDArray[Any]] = []
    logits_all: List[npt.NDArray[Any]] = []

    for classes in range(num_classes):
        pseudo_labels = generate_dirichlet(
            batch_size=num_images_per_class,
            class_id=classes,
            scale=scale,
            similarity_matrix=norm_similarity_matrix,
        )
        for i in range(num_images_per_class):
            rand_img = np.random.normal(0.5, 0.5, size=image_shape)
            rand_img = rand_img.astype(np.float32)
            rand_img_tensor = torch.tensor(
                rand_img.reshape((1, *reversed(image_shape))),
                requires_grad=True,
                device=device,
            )

            logits = model(rand_img_tensor)

            loss = criterion(logits, torch.tensor(pseudo_labels[i], device=device).reshape(1, -1))
            loss.backward()

            # TODO: add assert on weights

            synth_image = rand_img_tensor + rand_img_tensor.grad
            synth_logits = model(synth_image).detach().cpu()
            synth_image = synth_image.detach().cpu()

            images_all.append(synth_image)
            logits_all.append(synth_logits)

    return (np.array(images_all, dtype=torch.TensorType), np.array(logits_all, dtype=torch.TensorType))
