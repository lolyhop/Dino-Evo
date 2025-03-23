from nn.individual import Individual
from nn.genome import Genome
from typing import Callable
from nn.edge import Edge, Link
from nn.node import Node
import random
from settings import settings
import numpy as np
from nn.relu import relu
from typing import List


def crossover(dominant: Individual, recessive: Individual) -> Genome:
    offspring: Genome = Genome(
        dominant.genome.in_features,
        dominant.genome.out_features,
    )

    for node in dominant.genome.nodes:
        node_id: int = node.id
        node_recessive = recessive.genome.find_node(node_id)
        if not node_recessive:
            offspring.add_node(node)
        else:
            offspring.add_node(crossover_node(node, node_recessive))

    for edge in dominant.genome.edges:
        link: Link = edge.link
        edge_recessive = recessive.genome.find_edge(link)
        if not edge_recessive:
            offspring.add_edge(edge)
        else:
            offspring.add_edge(crossover_edge(edge, edge_recessive))

    return offspring


def crossover_node(a: Node, b: Node) -> Node:
    assert a.id == b.id
    node_id: int = a.id
    bias: float = random.choice([a.bias, b.bias])
    activation: Callable = random.choice([a.activation, b.activation])
    return Node(node_id, bias, activation)


def crossover_edge(a: Edge, b: Edge) -> Edge:
    assert a.link == b.link
    link: Link = a.link
    weight: float = random.choice([a.weight, b.weight])
    is_enabled: bool = random.choice([a.is_enabled, b.is_enabled])
    return Edge(link, weight, is_enabled)


def mutate(genome: Genome) -> None:
    if random.random() < settings.mutation_rate:
        mutate_add_edge(genome)
    if random.random() < settings.mutation_rate:
        mutate_add_node(genome)
    if random.random() < settings.mutation_rate:
        mutate_remove_node(genome)
    if random.random() < settings.mutation_rate:
        mutate_weights(genome)
    if random.random() < settings.mutation_rate:
        mutate_bias(genome)


def mutate_add_edge(genome: Genome) -> None:
    in_id_range: range = genome.get_hidden_nodes() + genome.get_output_nodes()
    if len(in_id_range) == 0:
        return

    input_id: int = random.choice(range(1, genome.in_features + 1))
    output_id: int = random.choice(in_id_range)
    while input_id == output_id:
        output_id = random.choice(in_id_range)
    new_link: Link = Link(input_id, output_id)

    # Check for duplicates
    existing_edge: Edge | None = genome.find_edge(new_link)
    if existing_edge is not None:
        existing_edge.is_enabled = True
        return

    if genome.would_create_cycle(new_link):
        return

    genome.add_edge(Edge(new_link, random.random(), True))


def mutate_add_node(genome: Genome):
    if not genome.edges:
        return
    old_edge: Edge = random.choice(genome.edges)
    old_edge.is_enabled = False

    new_node: Node = Node(genome.node_counter.increment(), random.random(), relu)

    genome.add_node(new_node)

    old_link: Link = old_edge.link
    old_weight: float = old_edge.weight

    genome.add_edge(Edge(Link(old_link.input_id, new_node.id), 1.0, True))
    genome.add_edge(Edge(Link(new_node.id, old_link.output_id), old_weight, True))


def mutate_remove_node(genome: Genome):
    hidden_nodes: List[int] = genome.get_hidden_nodes()
    if not hidden_nodes:
        return

    node_id: int = random.choice(hidden_nodes)
    genome.remove_node(node_id)


def mutate_weights(genome: Genome) -> None:
    for edge in genome.edges:
        if random.random() < settings.mutation_rate:
            edge.weight += np.random.normal(0, settings.mutation_scale)


def mutate_bias(genome: Genome) -> None:
    for node in genome.nodes:
        if random.random() < settings.mutation_rate:
            node.bias += np.random.normal(0, settings.mutation_scale)
