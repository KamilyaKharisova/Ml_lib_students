import numpy as np

def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """ Todo calculate loss of your model without loops"""
    diff = predictions - targets
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    # abs_diff = np.absolute(diff)
    # mean_diff = abs_diff.mean()
    return mean_diff