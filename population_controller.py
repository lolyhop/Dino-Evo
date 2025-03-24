import random
from typing import Any

from pygame import Surface

from dinosaur import Dinosaur
from dinosaur_controller import DinosaurController
from entities import Obstacle
from settings import settings

from nn.evolutionary_operators import crossover, mutate
from nn.genome import Genome

INPUT_FEATURES: int = 9
OUTPUT_FEATURES: int = 3


class PopulationController:
    def __init__(self) -> None:
        self.population_size: int = settings.population_size
        self.mutation_rate: float = settings.mutation_rate
        self.mutation_scale: float = settings.mutation_scale
        self.selection_amount: float = settings.selection_amount
        self.population: list[Dinosaur] = []
        self.previous_best_fitness: int = 0

    def check_population_alive(self) -> bool:
        """Check if any dinosaurs in the population are still alive."""
        return any(dinosaur.is_alive for dinosaur in self.population)

    def evolve_population(self) -> None:
        """
        Evolve the population by selecting the best dinosaurs and breeding them.
        Handles stagnation by introducing more diversity when needed.
        """
        # Track best fitness to detect stagnation
        current_best_fitness: int = (
            max(dinosaur.fitness for dinosaur in self.population)
            if self.population
            else 0
        )

        # Check if we need to handle stagnation
        fitness_improvement: int = current_best_fitness - self.previous_best_fitness
        is_stagnating: bool = (
            fitness_improvement
            <= settings.stagnation_threshold * self.previous_best_fitness
        )

        # Store current best fitness for next generation comparison
        self.previous_best_fitness = current_best_fitness

        # Select the best dinosaurs
        best_dinosaurs: list[Dinosaur] = self.roulette_wheel_selection(self.population)

        # Calculate how many new dinosaurs we need to create
        num_children_needed: int = self.population_size - len(best_dinosaurs)

        # Perform crossover on the best dinosaurs
        new_population: list[Dinosaur] = []
        while len(new_population) < num_children_needed:
            # Select two random parents from the best dinosaurs
            parent1, parent2 = random.sample(best_dinosaurs, 2)
            parent1_genome: Genome = parent1.dino_controller.genome
            parent2_genome: Genome = parent2.dino_controller.genome

            child_genome: Genome = crossover(parent1_genome, parent2_genome)

            # Increase mutation rate and scale if stagnating
            mutation_rate = self.mutation_rate * (3 if is_stagnating else 1)
            mutation_scale = self.mutation_scale * (2 if is_stagnating else 1)

            mutate(child_genome, mutation_rate, mutation_scale)
            new_population.append(Dinosaur(DinosaurController(child_genome)))

        # If stagnating, replace some of the population with completely new dinosaurs
        if is_stagnating:
            fresh_dinos_count: int = int(
                self.population_size * settings.stagnation_replacement_percentage
            )  # Replace with fresh dinosaurs
            new_genome = Genome(
                INPUT_FEATURES,
                OUTPUT_FEATURES,
            )
            new_genome.initialize_genome()
            new_population.extend(
                [
                    Dinosaur(DinosaurController(new_genome))
                    for _ in range(fresh_dinos_count)
                ]
            )

            # Remove 20% from the best dinosaurs
            best_dinosaurs: list[Dinosaur] = best_dinosaurs[:-fresh_dinos_count]

        self.population = best_dinosaurs + new_population

    def roulette_wheel_selection(self, population: list[Dinosaur]) -> list[Dinosaur]:
        """Select dinosaurs from the population using a roulette wheel selection method."""
        total_fitness: float = sum(dinosaur.fitness for dinosaur in population)
        probabilities: list[float] = [
            dinosaur.fitness / total_fitness for dinosaur in population
        ]
        return random.choices(
            population,
            weights=probabilities,
            k=int(len(population) * self.selection_amount),
        )

    def initialize_population(self) -> None:
        self.population = []
        for _ in range(self.population_size):
            new_genome = Genome(
                INPUT_FEATURES,
                OUTPUT_FEATURES,
            )
            new_genome.initialize_genome()
            self.population.append(Dinosaur(DinosaurController(new_genome)))

    def update_population(self, game_metadata: dict[str, Any]) -> None:
        for dinosaur in self.population:
            dinosaur.update(game_metadata)

    def check_collisions(self, obstacles: list[Obstacle]) -> None:
        """
        Check for collisions between dinosaurs and obstacles.

        When a dinosaur collides with an obstacle, it is marked as dead.

        Args:
            obstacles: list of obstacles to check for collisions
        """
        for dinosaur in self.population:
            for obstacle in obstacles:
                if dinosaur.dino_rect.colliderect(obstacle.rect):
                    dinosaur.is_alive = False

    def draw_population(self, screen: Surface) -> None:
        for dinosaur in self.population:
            dinosaur.draw(screen)
