import numpy as np


def relu(x: float) -> float:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
        x (float): The input value to the activation function.

    Returns:
        float: The output of the ReLU function, which is max(0, x).
    """
    return max(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Applies the softmax activation function to a numpy array.

    Args:
        x (np.ndarray): The input array of logits (raw prediction scores).

    Returns:
        np.ndarray: An array of probabilities corresponding to the input logits.
    """
    # Subtract the maximum value to prevent overflow
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)


ACTIVATION_MAP = {
    "relu": relu,
    "softmax": softmax,
}

REVERSE_ACTIVATION_MAP = {v: k for k, v in ACTIVATION_MAP.items()}
