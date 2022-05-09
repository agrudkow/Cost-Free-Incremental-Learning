from typing import Any
import numpy as np
import numpy.typing as npt


def generate_dirichlet(
    batch_size: int,
    class_id: int,
    scale: float,
    similarity_matrix: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    x = []
    sim = similarity_matrix[class_id, :]
    for _ in range(batch_size):
        temp = (sim - np.min(sim)) / (np.max(sim) - np.min(sim))
        temp = temp * scale + 0.0001
        x.append(np.random.dirichlet(temp))

    return np.array(x)
