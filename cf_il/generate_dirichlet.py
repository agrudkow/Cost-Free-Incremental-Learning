from typing import Any, Tuple
import numpy as np
import numpy.typing as npt


def generate_dirichlet(
    batch_size: int,
    class_id: int,
    scale: Tuple[float, float],
    similarity_matrix: npt.NDArray[Any],
    eta: float,
) -> npt.NDArray[Any]:
    beta = scale[0] if class_id < similarity_matrix.shape[0] // 2 else scale[1]
    x = []
    alpha = similarity_matrix[class_id, :]
    for _ in range(batch_size):
        i = 0
        while True:
            print(f'Generate Dirichlet iter: {i}')
            temp = alpha * beta + 1e-8
            sample = np.random.dirichlet(temp)
            sample_loss = np.square(np.linalg.norm(sample - alpha))
            if i > 1_000_000:
                raise Exception('Sample generation did not succeeded! Exceeded 1000000 iterations!')
            if sample_loss >= eta:
                print(f'Sample loss: {sample_loss}')
                i += 1
                continue
            x.append(sample)
            break

    return np.array(x)
