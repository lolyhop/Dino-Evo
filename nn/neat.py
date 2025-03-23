import random
import numpy as np
from nn.population import Population
from nn.ffn import FeedForwardNetwork
from nn.individual import Individual
from nn.genome import Genome


# TODO: complete the algorithm
class NEAT:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        population_size: int = 10,
        generations: int = 100,
    ):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.population_size: int = population_size
        self.generations: int = generations
        self.population: Population = Population(
            in_features, out_features, population_size
        )

    def run(self):
        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")

            # Evaluate fitness for each individual
            for individual in self.population.individuals:
                network = FeedForwardNetwork(individual.genome)
                fitness = self.evaluate_fitness(network)
                individual.fitness = fitness

            # Sort individuals by fitness
            sorted_individuals = self.population.sort_individuals_by_fitness(
                self.population.individuals
            )

            # Select the top individuals for the next generation
            next_generation = sorted_individuals[: self.population_size // 2]

            # Perform crossover and mutation to create the next generation
            while len(next_generation) < self.population_size:
                parent1 = random.choice(next_generation)
                parent2 = random.choice(next_generation)
                offspring_genome = self.population.crossover(parent1, parent2)

                # Mutate the offspring
                self.mutate(offspring_genome)

                # Create a new individual with the offspring genome
                offspring_individual = Individual(
                    offspring_genome, self.population.compute_fitness(offspring_genome)
                )
                next_generation.append(offspring_individual)

            # Update the population
            self.population.individuals = next_generation

        # Return the best individual after all generations
        best_individual = self.population.sort_individuals_by_fitness(
            self.population.individuals
        )[0]
        return best_individual

    def evaluate_fitness(self, network: FeedForwardNetwork) -> float:
        # TODO: Implement a proper fitness function based on the task
        # For example, if the task is to solve a classification problem, you could use accuracy as the fitness.
        # Here, we use a placeholder fitness function.
        inputs = np.random.rand(self.in_features)
        outputs = network.forward(inputs)
        return float(np.sum(outputs))  # Placeholder fitness function

    def mutate(self, genome: Genome):
        # Apply mutations to the genome
        if random.random() < 0.8:  # 80% chance to mutate weights
            self.mutate_weights(genome)
        if random.random() < 0.1:  # 10% chance to add a new node
            self.population.mutate_add_node(genome)
        if random.random() < 0.1:  # 10% chance to add a new edge
            self.population.mutate_add_edge(genome)
        if random.random() < 0.05:  # 5% chance to remove a node
            self.population.mutate_remove_node(genome)

    def mutate_weights(self, genome: Genome):
        # Mutate the weights of the edges in the genome
        for edge in genome.edges:
            if random.random() < 0.9:  # 90% chance to mutate the weight
                edge.weight += random.gauss(0, 0.1)  # Add Gaussian noise to the weight
