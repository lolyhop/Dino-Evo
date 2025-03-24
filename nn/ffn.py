import numpy as np
from nn.genome import Genome
from nn.node import Node
from nn.edge import Edge


class FeedForwardNetwork:
    def __init__(self, genome: Genome):
        self.genome: Genome = genome
        self.nodes: list[Node] = genome.nodes
        self.edges: list[Edge] = genome.edges
        self.input_size: int = genome.in_features
        self.output_size: int = genome.out_features

        # Create a mapping from node ID to node object
        self.node_map: dict[int, Node] = {node.id: node for node in self.nodes}

    def _get_topological_order(self) -> list[int]:
        """
        Computes the topological order of the nodes in the feedforward network.

        This method ensures that each node is processed only after all its dependencies (input nodes)
        have been processed. It starts with the input nodes and iteratively adds nodes to the order
        as their dependencies are satisfied. If a cycle is detected, the remaining nodes are added
        in an arbitrary order to prevent an infinite loop.

        Returns:
            list[int]: A list of node IDs in topological order.
        """
        # Start with input nodes
        order = list(range(1, self.input_size + 1))

        # Create a dictionary of dependencies (which nodes need to be processed before others)
        dependencies: dict[int, list[int]] = {node.id: [] for node in self.nodes}
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

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feedforward network.

        This method takes an input array, processes it through the network, and returns the output values
        from the output nodes. It computes the topological order of the nodes to ensure that each node is
        processed only after all its dependencies have been satisfied. The method also handles the activation
        of each node based on the weighted sum of its inputs and its bias.

        Args:
            inputs (np.ndarray): A numpy array containing the input values for the network.

        Returns:
            np.ndarray: A numpy array containing the output values from the output nodes.
        """
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
