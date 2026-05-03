from game.entities import Obstacle, SmallCactus, LargeCactus, Bird
from game.population_controller import PopulationController
from utils.serialization import deserialize_population, serialize_population
from torch.utils.tensorboard import SummaryWriter
from settings import settings
from typing import Any
import random
import os
import time
import numpy as np


class HeadlessChromeDinoGame:
    """
    Headless version of the Chrome Dino Game for faster training.
    This version removes all rendering to reduce computational resources.
    """

    def __init__(self) -> None:
        # Initialize game state
        self.game_speed: int = settings.game_speed
        self.max_generation_time: int = settings.max_generation_time
        self.points: int = 0
        self.obstacles: list[Obstacle] = []

        # Setup TensorBoard writer
        log_dir = os.path.join("logs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print("Run 'tensorboard --logdir=logs' to view training progress")

        if os.path.exists(settings.serialization_path):
            self.population_controller: PopulationController = PopulationController(
                deserialize_population(settings.serialization_path)
            )
        else:
            self.population_controller: PopulationController = PopulationController()

    def _log_statistics(self, n_generation: int, start_time: float) -> None:
        """
        Log training statistics to TensorBoard.

        Args:
            n_generation: Current generation number
            start_time: Training start time
        """
        # Calculate fitness statistics
        fitness_values = [
            dino.fitness for dino in self.population_controller.population
        ]
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        median_fitness = np.median(fitness_values)
        fitness_std = np.std(fitness_values)

        # Calculate time statistics
        current_time = time.time()
        elapsed_time = current_time - start_time
        gens_per_second = (n_generation + 1) / elapsed_time if elapsed_time > 0 else 0

        # Log fitness metrics
        self.writer.add_scalar("Fitness/Best", max_fitness, n_generation)
        self.writer.add_scalar("Fitness/Average", avg_fitness, n_generation)
        self.writer.add_scalar("Fitness/Median", median_fitness, n_generation)
        self.writer.add_scalar("Fitness/StdDev", fitness_std, n_generation)

        # Log game metrics
        self.writer.add_scalar("Game/Score", self.points, n_generation)
        self.writer.add_scalar("Game/Speed", self.game_speed, n_generation)

        # Log training metrics
        self.writer.add_scalar(
            "Training/GenerationsPerSecond", gens_per_second, n_generation
        )

        # Create histogram of fitness values
        self.writer.add_histogram(
            "Fitness/Distribution", np.array(fitness_values), n_generation
        )

    def run(self) -> None:
        """Main game loop that handles game state updates without rendering."""

        if len(self.population_controller.population) == 0:
            self.population_controller.initialize_population()

        n_generation: int = 0
        start_time = time.time()

        try:
            while True:
                # Track start time for this generation
                gen_start_time = time.time()

                while True:
                    # Compose metadata about the current state of the game
                    game_metadata: dict[str, Any] = {
                        "points": self.points,
                        "game_speed": self.game_speed,
                        "obstacles": self.obstacles,
                    }

                    self.population_controller.update_population(game_metadata)

                    # Generate obstacles
                    if len(self.obstacles) == 0:
                        obstacle_type: int = random.randint(0, 2)
                        if obstacle_type == 0:
                            self.obstacles.append(SmallCactus())
                        elif obstacle_type == 1:
                            self.obstacles.append(LargeCactus())
                        elif obstacle_type == 2:
                            self.obstacles.append(Bird())

                    # Update obstacles
                    for obstacle in self.obstacles:
                        obstacle.update(self.game_speed)

                        # Remove obstacles that are off-screen
                        if obstacle.rect.x < -obstacle.rect.width:
                            self.obstacles.remove(obstacle)

                    # Check for collisions with dinosaurs
                    self.population_controller.check_collisions(self.obstacles)

                    # Update score and game speed
                    self.points += 1
                    if self.points % 10 == 0:
                        self.game_speed += settings.game_acceleration
                        self.game_speed = min(self.game_speed, settings.max_game_speed)

                    generation_time = time.time() - gen_start_time

                    # Check termination conditions: population dead, time limit
                    if (
                        not self.population_controller.check_population_alive()
                        or generation_time >= self.max_generation_time
                    ):
                        break

                # Log statistics to TensorBoard
                self._log_statistics(n_generation, start_time)

                # Reinitialize population
                n_generation += 1
                self.population_controller.evolve_population()

                # Reset game state
                self.game_speed = settings.game_speed
                self.points = 0
                self.obstacles = []

                # Periodically save population and adjust max generation time
                if n_generation % 10 == 0:
                    serialize_population(
                        [
                            dinosaur.dino_controller.genome
                            for dinosaur in self.population_controller.population
                        ],
                        settings.serialization_path,
                    )
                    self.max_generation_time += random.randint(0, 2)

        except KeyboardInterrupt:
            # Log final statistics before exiting
            print("\nTraining interrupted. Saving population...")
            self._log_statistics(n_generation, start_time)

            # Save population on exit
            serialize_population(
                [
                    dinosaur.dino_controller.genome
                    for dinosaur in self.population_controller.population
                ],
                settings.serialization_path,
            )

            # Close TensorBoard writer
            self.writer.close()


def main() -> None:
    """Main function to run the headless game"""
    print("Starting Headless Chrome Dino Game for faster training...")
    print("Press Ctrl+C to stop training and save the population.")
    game = HeadlessChromeDinoGame()
    game.run()


if __name__ == "__main__":
    main()
