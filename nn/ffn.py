import numpy as np
from nn.genome import Genome
from nn.node import Node
from nn.edge import Edge
from typing import List


# TODO: verify correctness and integrate usage of bias/activation
class FeedForwardNetwork:
    def __init__(self, genome: Genome):
        self.genome: Genome = genome
        self.nodes: List[Node] = genome.nodes
        self.edges: List[Edge] = genome.edges
        self.input_size: int = genome.in_features
        self.output_size: int = genome.out_features

    def forward(self, inputs):
        # Create a dictionary to store the output of each node
        node_outputs = {node.id: 0.0 for node in self.nodes}

        # Initialize input nodes
        for i in range(self.input_size):
            node_outputs[i] = inputs[i]

        # Process the network in a feedforward manner
        for edge in self.edges:
            if edge.is_enabled:
                input_node = edge.link.input_id
                output_node = edge.link.output_id
                weight = edge.weight

                # Apply the activation function to the input node's output
                input_value = node_outputs[input_node]
                output_value = self.nodes[output_node].activation(input_value * weight)

                # Accumulate the output value
                node_outputs[output_node] += output_value

        # Collect the output values from the output nodes
        outputs = []
        for i in range(self.input_size, self.input_size + self.output_size):
            outputs.append(node_outputs[i])

        return np.array(outputs)
