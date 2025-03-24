from typing import Callable


class Node:
    def __init__(self, node_id: int, bias: float, activation: Callable):
        self.id: int = node_id
        self.bias: float = bias
        self.activation: Callable = activation
