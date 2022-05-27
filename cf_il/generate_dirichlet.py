from typing import Any, Tuple
import numpy as np
import numpy.typing as npt


def np_softmax(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.exp(x) / sum(np.exp(x))


def generate_dirichlet(
    batch_size: int,
    class_id: int,
    scale: Tuple[float, float],
    similarity_matrix: npt.NDArray[Any],
    eta: float,
    max_iter: int,
) -> npt.NDArray[Any]:
    beta = scale[0] if class_id < similarity_matrix.shape[0] // 2 else scale[1]
    x = []
    alpha = similarity_matrix[class_id, :]
    for j in range(batch_size):
        i = 0
        alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
        temp = alpha * beta + 1e-8  # Add small number to prevent from being 0
        while True:
            print(f'Generate Dirichlet sample {j + 1}/{batch_size} - iter: {i}')
            sample = np.random.dirichlet(temp)
            sample_loss = np.square(np.linalg.norm(sample - np_softmax(alpha)))
            print(f'Sample loss: {sample_loss}')
            if i > max_iter:
                raise Exception(f'Sample generation did not succeeded! Exceeded {max_iter} iterations!')
            if sample_loss >= eta:
                i += 1
                continue
            x.append(sample)
            break

    return np.array(x)
