import numpy as np
from nn.genome import Genome
from nn.node import Node
from nn.edge import Edge
from typing import List, Dict


# TODO: verify correctness and integrate usage of bias/activation
class FeedForwardNetwork:
    def __init__(self, genome: Genome):
        self.genome: Genome = genome
        self.nodes: List[Node] = genome.nodes
        self.edges: List[Edge] = genome.edges
        self.input_size: int = genome.in_features
        self.output_size: int = genome.out_features

        # Create a mapping from node ID to node object
        self.node_map: Dict[int, Node] = {node.id: node for node in self.nodes}

    def _get_topological_order(self) -> List[int]:
        """Compute topological ordering of nodes for correct forward pass."""
        # Start with input nodes
        order = list(range(1, self.input_size + 1))

        # Create a dictionary of dependencies (which nodes need to be processed before others)
        dependencies: Dict[int, List[int]] = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            if not edge.is_enabled:
                continue
            # TODO: Understand why edges that should be deleted are still in the list (Recheck Genome.remove_node() and mutate_remove_node)
            if edge.link.output_id not in dependencies:
                continue
            dependencies[edge.link.output_id].append(edge.link.input_id)

        # Add nodes that have all dependencies satisfied
        visited = set(order)
        remaining = set(node.id for node in self.nodes) - visited

        while remaining:
            progress = False
            for node_id in list(remaining):
                # Check if all dependencies are in the visited set
                if all(dep in visited for dep in dependencies[node_id]):
                    order.append(node_id)
                    visited.add(node_id)
                    remaining.remove(node_id)
                    progress = True

            # If no progress was made in this iteration, there might be a cycle
            if not progress:
                # Add remaining nodes in some order (sub-optimal but prevents infinite loop)
                order.extend(list(remaining))
                break

        return order

    def forward(self, inputs):
        # Create dictionaries to store the output of each node
        node_outputs = {node.id: 0.0 for node in self.nodes}

        # Initialize input nodes
        for i in range(1, self.input_size + 1):
            node_outputs[i] = inputs[i - 1]

        # Get topological ordering for correct processing
        node_order = self._get_topological_order()

        # Skip input nodes that are already processed
        node_order = node_order[self.input_size :]

        # Process nodes in topological order
        for node_id in node_order:
            # Use node_map to get node by ID instead of using index
            node = self.node_map[node_id]

            # Compute the weighted sum of inputs to this node
            weighted_sum = 0.0
            for edge in self.edges:
                if edge.is_enabled and edge.link.output_id == node_id:
                    input_node_id = edge.link.input_id
                    weighted_sum += node_outputs[input_node_id] * edge.weight

            # Add bias if it exists in the node
            if hasattr(node, "bias"):
                weighted_sum += node.bias

            # Apply activation function to the sum
            node_outputs[node_id] = node.activation(weighted_sum)

        # Collect the output values from the output nodes
        outputs = []
        output_ids = range(self.input_size + 1, self.input_size + self.output_size + 1)
        for i in output_ids:
            outputs.append(node_outputs[i])

        return np.array(outputs)
