from typing import Any
import numpy as np
import numpy.typing as npt


def generate_dirichlet(
    batch_size: int,
    class_id: int,
    scale: float,
    similarity_matrix: npt.NDArray[Any],
    eta: float,
) -> npt.NDArray[Any]:
    x = []
    sim = similarity_matrix[class_id, :]
    for _ in range(batch_size):
        i = 0
        while True:
            print(f'Generate dirichlet iter: {i}')
            temp = (sim - np.min(sim)) / (np.max(sim) - np.min(sim))
            temp = temp * scale + 0.0001
            sample = np.random.dirichlet(temp)
            sample_loss = np.linalg.norm(sample - sim)
            if sample_loss >= eta:
                print(f'Sample loss: {sample_loss}')
                i += 1
                continue
            x.append(sample)
            break

    return np.array(x)
