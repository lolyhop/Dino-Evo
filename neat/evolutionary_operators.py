from neat.genome import Genome
from typing import Callable
from neat.edge import Edge, Link
from neat.node import Node
import random
import numpy as np
from neat.activations import relu


def crossover(dominant: Genome, recessive: Genome) -> Genome:
    """
    Performs crossover between two genomes to produce an offspring genome.

    Performs crossover on shared nodes and edges.
    Takes excessive nodes and edges from dominant parent.

    Args:
        dominant (Genome): The dominant genome to be crossed over.
        recessive (Genome): The recessive genome to be crossed over.

    Returns:
        Genome: A new genome that is the result of the crossover between the dominant and recessive genomes.
    """
    offspring: Genome = Genome(
        dominant.in_features,
        dominant.out_features,
    )

    for node in dominant.nodes:
        node_id: int = node.id
        node_recessive = recessive.find_node(node_id)
        if not node_recessive:
            offspring.add_node(node)
        else:
            offspring.add_node(crossover_node(node, node_recessive))

    for edge in dominant.edges:
        link: Link = edge.link
        edge_recessive = recessive.find_edge(link)
        if not edge_recessive:
            offspring.add_edge(edge)
        else:
            offspring.add_edge(crossover_edge(edge, edge_recessive))

    return offspring


def crossover_node(a: Node, b: Node) -> Node:
    """
    Performs crossover between two nodes to produce a new node.

    This function takes two nodes as input and randomly selects their bias and activation function
    to create a new node. The new node will have the same ID as the input nodes.

    Args:
        a (Node): The first node to crossover.
        b (Node): The second node to crossover.

    Returns:
        Node: A new node that is the result of the crossover between the two input nodes.
    """
    assert a.id == b.id
    node_id: int = a.id
    bias: float = random.choice([a.bias, b.bias])
    activation: Callable = random.choice([a.activation, b.activation])
    return Node(node_id, bias, activation)


def crossover_edge(a: Edge, b: Edge) -> Edge:
    """
    Performs crossover between two edges to produce a new edge.

    This function takes two edges as input and randomly selects their weight and enabled status
    to create a new edge. The new edge will have the same link as the input edges.

    Args:
        a (Edge): The first edge to crossover.
        b (Edge): The second edge to crossover.

    Returns:
        Edge: A new edge that is the result of the crossover between the two input edges.
    """
    assert a.link == b.link
    link: Link = a.link
    weight: float = random.choice([a.weight, b.weight])
    is_enabled: bool = random.choice([a.is_enabled, b.is_enabled])
    return Edge(link, weight, is_enabled)


def mutate(
    genome: Genome,
    mutation_rate: float,
    mutation_scale: float,
) -> None:
    """
    Mutates the given genome based on the specified mutation rate and scale.

    This function applies various mutation operations to the genome, including adding edges,
    adding nodes, removing nodes, and mutating weights and biases. The mutation operations
    are performed with a probability defined by the mutation_rate parameter.

    Args:
        genome (Genome): The genome to be mutated.
        mutation_rate (float): The probability of applying a mutation operation.
        mutation_scale (float): The scale factor for weight and bias mutations.

    Returns:
        None: This function modifies the genome in place.
    """
    if random.random() < mutation_rate:
        mutate_add_edge(genome)
    if random.random() < mutation_rate:
        mutate_add_node(genome)
    if random.random() < mutation_rate:
        mutate_remove_node(genome)
    mutate_weights(genome, mutation_scale, mutation_rate)
    mutate_bias(genome, mutation_scale, mutation_rate)


def mutate_add_edge(genome: Genome) -> None:
    """
    Adds a new edge to the genome by creating a link between two nodes.

    This function randomly selects an input node and an output node from the genome,
    ensuring that they are not the same. It then creates a new link and checks for
    duplicates before adding the edge to the genome. If the new edge would create a cycle,
    it is not added.

    Args:
        genome (Genome): The genome to which the new edge will be added.

    Returns:
        None: This function modifies the genome in place.
    """
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
    """
    Mutates the genome by adding a new node to an existing edge.

    This function selects a random edge from the genome, disables it, and creates a new node.
    The new node is connected to the input and output of the old edge, effectively splitting
    the edge into two. The new node's activation function is set to ReLU, and its weight is
    initialized randomly.

    Args:
        genome (Genome): The genome to which the new node will be added.

    Returns:
        None: This function modifies the genome in place.
    """
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
    """
    Mutates the genome by removing a node from the network.

    This function selects a random hidden node from the genome and removes it,
    effectively altering the structure of the neural network. If there are no
    hidden nodes available, the function will exit without making any changes.

    Args:
        genome (Genome): The genome from which a node will be removed.

    Returns:
        None: This function modifies the genome in place.
    """
    hidden_nodes: list[int] = genome.get_hidden_nodes()
    if not hidden_nodes:
        return

    node_id: int = random.choice(hidden_nodes)
    genome.remove_node(node_id)


def mutate_weights(
    genome: Genome,
    mutation_rate: float,
    mutation_scale: float,
) -> None:
    """
    Mutates the weights of the edges in the genome.

    This function iterates through all edges in the genome and applies a mutation
    to the weight of each edge based on the specified mutation rate and mutation scale.
    If a randomly generated number is less than the mutation rate, the weight of the edge
    is adjusted by adding a value drawn from a normal distribution with mean 0 and
    standard deviation equal to the mutation scale.

    Args:
        genome (Genome): The genome whose edges' weights will be mutated.
        mutation_rate (float): The probability of mutating each edge's weight.
        mutation_scale (float): The standard deviation of the normal distribution used for mutation.

    Returns:
        None: This function modifies the genome in place.
    """
    for edge in genome.edges:
        if random.random() < mutation_rate:
            edge.weight += np.random.normal(0, mutation_scale)


def mutate_bias(
    genome: Genome,
    mutation_rate: float,
    mutation_scale: float,
) -> None:
    """
    Mutates the biases of the nodes in the genome.

    This function iterates through all nodes in the genome and applies a mutation
    to the bias of each node based on the specified mutation rate and mutation scale.
    If a randomly generated number is less than the mutation rate, the bias of the node
    is adjusted by adding a value drawn from a normal distribution with mean 0 and
    standard deviation equal to the mutation scale.

    Args:
        genome (Genome): The genome whose nodes' biases will be mutated.
        mutation_rate (float): The probability of mutating each node's bias.
        mutation_scale (float): The standard deviation of the normal distribution used for mutation.

    Returns:
        None: This function modifies the genome in place.
    """
    for node in genome.nodes:
        if random.random() < mutation_rate:
            node.bias += np.random.normal(0, mutation_scale)
