import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return round(np.sum((targets - predictions) ** 2) / len(predictions), 2)
