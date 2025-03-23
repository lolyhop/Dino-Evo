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
        self.__initialize_genome(in_features, out_features)

    def __initialize_genome(self, in_features: int, out_features: int) -> None:
        for _ in range(in_features):
            new_node: Node = Node(self.node_counter.increment(), 0, relu)
            self.add_node(new_node)

        for _ in range(out_features):
            new_node: Node = Node(self.node_counter.increment(), 0, relu)
            self.add_node(new_node)

        for i in range(1, in_features + 1):
            for j in range(1, out_features + 1):
                self.add_edge(
                    Edge(
                        Link(i, in_features + j),
                        random.random() * np.sqrt(2 / in_features),
                        True,
                    )
                )

    def find_node(self, node_id: int) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def add_node(self, new_node: Node) -> None:
        self.nodes.append(new_node)

    def find_edge(self, link: Link) -> Edge:
        for edge in self.edges:
            if edge.link == link:
                return edge
        return None

    def add_edge(self, new_edge: Edge) -> None:
        self.edges.append(new_edge)

    def remove_node(self, node_id: int) -> None:
        node: Node = self.find_node(node_id)
        self.nodes.remove(node)

        to_remove: List[Edge] = []
        for edge in self.edges:
            if edge.link.input_id == node_id or edge.link.output_id == node_id:
                to_remove.append(edge)
        self.edges = [edge for edge in self.edges if edge not in to_remove]

    def get_input_or_hidden_nodes(self) -> List[int]:
        node_ids: Set[int] = set()
        for edge in self.edges:
            node_ids.add(edge.link.input_id)
        return list(node_ids)

    def get_output_nodes(self) -> List[int]:
        input_or_hidden_nodes: List[int] = self.get_input_or_hidden_nodes()
        node_ids: List[int] = [node.id for node in self.nodes]
        return list(set(node_ids) - set(input_or_hidden_nodes))

    def get_hidden_nodes(self) -> List[int]:
        hidden: List[int] = []
        for node in self.nodes:
            has_input: bool = False
            has_output: bool = False
            for edge in self.edges:
                if edge.link.input_id == node.id:
                    has_output = True
                if edge.link.output_id == node.id:
                    has_input = True
            if has_input and has_output:
                hidden.append(node.id)
        return hidden

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
