import random
import numpy as np
from nn.activations import relu
from nn.node import Node
from nn.edge import Edge, Link
from nn.counter import Counter


class Genome:
    def __init__(self, in_features: int, out_features: int):
        self.node_counter: Counter = Counter()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def initialize_genome(self) -> None:
        """
        Initializes the genome by creating input and output nodes, and connecting them with edges.

        This method populates the genome with nodes corresponding to the input features and output features.
        Each input node is initialized with a bias of 0 and a ReLU activation function. Edges are created
        between each input node and each output node with random weights, ensuring that the network is ready
        for use in a feedforward neural network.

        Returns:
            None: This method modifies the genome in place
        """
        for _ in range(self.in_features):
            new_node: Node = Node(self.node_counter.increment(), 0, relu)
            self.add_node(new_node)

        for _ in range(self.out_features):
            new_node: Node = Node(self.node_counter.increment(), 0, relu)
            self.add_node(new_node)

        for i in range(1, self.in_features + 1):
            for j in range(1, self.out_features + 1):
                self.add_edge(
                    Edge(
                        Link(i, self.in_features + j),
                        random.random() * np.sqrt(2 / self.in_features),
                        True,
                    )
                )

    def find_node(self, node_id: int) -> Node | None:
        """
        Finds a node in the genome by its ID.

        This method searches through the list of nodes in the genome and returns the node
        that matches the given node ID. If no node with the specified ID is found, it returns None.

        Args:
            node_id (int): The ID of the node to be found.

        Returns:
            Node | None: The node with the specified ID if found, otherwise None.
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def add_node(self, new_node: Node) -> None:
        """
        Adds a new node to the genome.

        This method appends the provided new_node to the list of nodes in the genome.
        It ensures that the genome's structure is updated to include the new node,
        which can represent an input, output, or hidden node in the neural network.

        Args:
            new_node (Node): The node to be added to the genome.

        Returns:
            None: This method modifies the genome in place.
        """
        self.nodes.append(new_node)

    def find_edge(self, link: Link) -> Edge | None:
        """
        Finds an edge in the genome by its link.

        This method searches through the list of edges in the genome and returns the edge
        that matches the given link. If no edge with the specified link is found, it returns None.

        Args:
            link (Link): The link of the edge to be found.

        Returns:
            Edge | None: The edge with the specified link if found, otherwise None.
        """
        for edge in self.edges:
            if edge.link == link:
                return edge
        return None

    def add_edge(self, new_edge: Edge) -> None:
        """
        Adds a new edge to the genome.

        This method appends the provided new_edge to the list of edges in the genome.
        It ensures that the genome's structure is updated to include the new edge,
        which represents a connection between nodes in the neural network.

        Args:
            new_edge (Edge): The edge to be added to the genome.

        Returns:
            None: This method modifies the genome in place.
        """
        self.edges.append(new_edge)

    def remove_node(self, node_id: int) -> None:
        """
        Removes a node from the genome.

        This method deletes the specified node from the genome's list of nodes and also
        removes any edges that are connected to this node. It ensures that the genome's
        structure remains consistent after the removal of the node.

        Args:
            node_id (int): The identifier of the node to be removed.

        Returns:
            None: This method modifies the genome in place.
        """
        node: Node | None = self.find_node(node_id)
        if node is None:
            return

        self.nodes.remove(node)

        edges_to_remove = []
        for edge in self.edges:
            if edge.link.input_id == node_id or edge.link.output_id == node_id:
                edges_to_remove.append(edge)

        # Remove the collected edges
        for edge in edges_to_remove:
            self.edges.remove(edge)

    def get_input_or_hidden_nodes(self) -> list[int]:
        """
        Retrieves a list of input and hidden node identifiers.

        This method combines the identifiers of input nodes and hidden nodes
        in the genome. It is useful for accessing all nodes that are not output
        nodes, which can be important for various operations in the neural network.

        Returns:
            list[int]: A list containing the identifiers of input and hidden nodes.
        """
        return self.get_input_nodes() + self.get_hidden_nodes()

    def get_input_nodes(self) -> list[int]:
        """
        Retrieves a list of input nodes.

        This method generates a list containing the identifiers of all input nodes
        in the genome. Input nodes are essential for feeding data into the neural
        network, and this method provides a convenient way to access them.

        Returns:
            list[int]: A list containing the identifiers of input nodes.
        """
        return list(range(1, self.in_features + 1))

    def get_output_nodes(self) -> list[int]:
        """
        Retrieves a list of output nodes.

        This method generates a list containing the identifiers of all output nodes
        in the genome. Output nodes are crucial for producing the final results of
        the neural network, and this method provides a convenient way to access them.

        Returns:
            list[int]: A list containing the identifiers of output nodes.
        """
        return list(
            range(self.in_features + 1, self.in_features + self.out_features + 1)
        )

    def get_hidden_nodes(self) -> list[int]:
        """
        Retrieves a list of hidden nodes.

        This method generates a list containing the identifiers of all hidden nodes
        in the genome. Hidden nodes are important for the internal processing of the
        neural network, as they allow for complex representations and transformations
        of the input data. This method provides a convenient way to access them.

        Returns:
            list[int]: A list containing the identifiers of hidden nodes.
        """
        return list(
            range(self.in_features + self.out_features + 2, len(self.nodes) + 1)
        )

    def would_create_cycle(self, new_link: Link) -> bool:
        """
        Checks if adding a new link would create a cycle in the genome.

        This method traverses the existing edges in the genome to determine
        if the new link, defined by its input and output IDs, would result
        in a cycle. A cycle occurs when there is a path from the output node
        back to the input node, which can lead to infinite loops during
        the forward pass of the neural network.

        Args:
            new_link (Link): The link to be checked for potential cycles.

        Returns:
            bool: True if adding the new link would create a cycle, False otherwise.
        """
        input_id: int = new_link.output_id
        while True:
            stop_flag: bool = True
            for edge in self.edges:
                if edge.link.input_id == input_id:
                    if edge.link.output_id == new_link.input_id:
                        return True
                    input_id = edge.link.output_id
                    stop_flag = False
                    break
            if stop_flag:  # No further connections
                return False
