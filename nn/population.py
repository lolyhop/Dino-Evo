from nn.individual import Individual
from nn.genome import Genome
from nn.counter import Counter
from typing import Callable
from nn.edge import Edge, Link
from nn.node import Node
import random
import numpy as np
from typing import List


class Population:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        population_size: int = 10,
    ):
        self.genome_counter: Counter = Counter()
        self.node_counter: Counter = Counter()
        self.edge_counter: Counter = Counter()
        self.individuals: List[Individual] = []

        for _ in range(population_size):
            new_genome: Genome = self.get_new_genome(in_features, out_features)
            self.individuals.append(
                Individual(new_genome, self.compute_fitness(new_genome))
            )

    def compute_fitness(self, genome: Genome) -> float:
        """Evaluate the fitness of an individual (placeholder)."""
        # TODO: insert fitness function instead of random
        return random.random()  # Example fitness function

    def sort_individuals_by_fitness(
        self, individuals: List[Individual]
    ) -> List[Individual]:
        """Sort individuals by fitness in descending order."""
        return sorted(individuals, key=lambda ind: ind.fitness, reverse=True)

    def get_new_genome(self, in_features: int, out_features: int) -> Genome:
        # Generate primitive fully-connected FFN
        genome: Genome = Genome(
            self.genome_counter.increment(), in_features, out_features
        )

        # TODO: add proper weights and activation function initialization
        #       for new nodes and edges
        for _ in range(out_features):
            new_node: Node = Node(
                self.node_counter.increment(), random.random(), lambda x: 1 / np.exp(x)
            )
            genome.add_node(new_node)

        for i in range(in_features):
            new_node: Node = Node(
                self.node_counter.increment(), random.random(), lambda x: 1 / np.exp(x)
            )
            for j in range(out_features):
                input_id: int = new_node.id
                output_id: int = genome.nodes[j].id
                new_edge: Edge = Edge(Link(input_id, output_id), 1.0, True)
                genome.add_edge(new_edge)

        return genome

    def crossover(self, dominant: Individual, recessive: Individual) -> Genome:
        offspring: Genome = Genome(
            self.genome_counter.increment(),
            dominant.genome.in_features,
            dominant.genome.out_features,
        )

        for node in dominant.genome.nodes:
            node_id: int = node.id
            node_recessive = recessive.genome.find_node(node_id)
            if not node_recessive:
                offspring.add_node(node)
            else:
                offspring.add_node(self.crossover_node(node, node_recessive))

        for edge in dominant.genome.edges:
            link: Link = edge.link
            edge_recessive = recessive.genome.find_edge(link)
            if not edge_recessive:
                offspring.add_edge(edge)
            else:
                offspring.add_edge(self.crossover_edge(edge, edge_recessive))

        return offspring

    def crossover_node(self, a: Node, b: Node) -> Node:
        assert a.id == b.id
        node_id: int = a.id
        bias: float = random.choice([a.bias, b.bias])
        activation: Callable = random.choice([a.activation, b.activation])
        return Node(node_id, bias, activation)

    def crossover_edge(self, a: Edge, b: Edge) -> Edge:
        assert a.link == b.link
        link: Link = a.link
        weight: float = random.choice([a.weight, b.weight])
        is_enabled: bool = random.choice([a.is_enabled, b.is_enabled])
        return Edge(link, weight, is_enabled)

    # TODO: integrate these mutations
    def mutate_add_edge(self, genome: Genome) -> None:
        input_id: int = random.choice(genome.get_input_or_hidden_nodes())
        output_id: int = random.choice(genome.get_output_nodes())
        new_link: Link = Link(input_id, output_id)

        # Check for duplicates
        existing_edge: Edge = genome.find_edge(new_link)
        if existing_edge:
            existing_edge.is_enabled = True
            return

        if genome.would_create_cycle(new_link):
            return

        # TODO: improve weight mutation
        genome.add_edge(Edge(new_link, random.random(), True))

    def mutate_add_node(self, genome: Genome):
        if not genome.edges:
            return
        old_edge: Edge = random.choice(genome.edges)
        old_edge.is_enabled = False

        # TODO: improve bias mutation
        # TODO: change activation function definition
        new_node: Node = Node(
            self.node_counter.increment(), random.random(), lambda x: 1 / np.exp(x)
        )

        old_link: Link = old_edge.link
        old_weight: float = old_edge.weight

        genome.add_edge(Edge(Link(old_link.input_id, new_node.id), 1.0, True))
        genome.add_edge(Edge(Link(new_node.id, old_link.output_id), old_weight, True))

    def mutate_remove_node(self, genome: Genome):
        hidden_nodes: List[int] = genome.get_hidden_nodes()
        if not hidden_nodes:
            return

        node_id: int = random.choice(hidden_nodes)
        genome.remove_node(node_id)

    # TODO: add simple, non-structural mutations over weights and bias
