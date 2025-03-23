import random
import numpy as np
from typing import List, Set
from nn.relu import relu
from nn.node import Node
from nn.edge import Edge, Link
from nn.counter import Counter


class Genome:
    def __init__(self, in_features: int, out_features: int):
        self.node_counter: Counter = Counter()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

    def initialize_genome(self) -> None:
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
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def add_node(self, new_node: Node) -> None:
        self.nodes.append(new_node)

    def find_edge(self, link: Link) -> Edge | None:
        for edge in self.edges:
            if edge.link == link:
                return edge
        return None

    def add_edge(self, new_edge: Edge) -> None:
        self.edges.append(new_edge)

    def remove_node(self, node_id: int) -> None:
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

    def get_input_or_hidden_nodes(self) -> List[int]:
        return self.get_input_nodes() + self.get_hidden_nodes()

    def get_input_nodes(self) -> List[int]:
        return list(range(1, self.in_features + 1))

    def get_output_nodes(self) -> List[int]:
        return list(
            range(self.in_features + 1, self.in_features + self.out_features + 1)
        )

    def get_hidden_nodes(self) -> List[int]:
        return list(
            range(self.in_features + self.out_features + 2, len(self.nodes) + 1)
        )

    def would_create_cycle(self, new_link: Link) -> bool:
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
