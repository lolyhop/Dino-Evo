import numpy as np


def relu(x: float) -> float:
    return max(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    # Subtract the maximum value to prevent overflow
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)
