from __future__ import annotations
import random
from typing import List, Dict, Any

import pygame
from pygame import Surface
from pygame.time import Clock

from settings import settings
from entities import Obstacle, Cloud, SmallCactus, LargeCactus, Bird, Background
from population_controller import PopulationController


class ChromeDinoGame:
    """
    Main class for the Chrome Dino Game.
    """

    def __init__(self) -> None:
        # Initialize pygame
        pygame.init()
        settings.initialize_font()

        # Setup display
        self._setup_display()

        # Initialize game state
        self.game_speed: int = settings.game_speed
        self.x_pos_bg: int = 0
        self.y_pos_bg: int = 380
        self.points: int = 0
        self.obstacles: List[Obstacle] = []
        self.population_controller: PopulationController = PopulationController()

    def run(self) -> None:
        """Main game loop that handles events, updates game state, and renders."""

        # Create game objects
        cloud = Cloud()
        background = Background()
        clock = Clock()

        # Initialize population of dinosaurs
        self.population_controller.initialize_population()

        running: bool = True
        n_generation: int = 0
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Clear screen
            self.screen.fill((255, 255, 255))

            # Compose metadata about the current state of the game
            game_metadata: Dict[str, Any] = {
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

            # Update and draw obstacles
            for obstacle in self.obstacles:
                obstacle.update(self.game_speed)
                obstacle.draw(self.screen)

                # Remove obstacles that are off-screen
                if obstacle.rect.x < -obstacle.rect.width:
                    self.obstacles.remove(obstacle)

            # Check for collisions with dinosaurs
            self.population_controller.check_collisions(self.obstacles)

            # Update and draw background
            background.update(self.screen, self.game_speed)
            background.draw(self.screen)

            # Update and draw cloud
            cloud.update(self.game_speed)
            cloud.draw(self.screen)

            self.population_controller.draw_population(self.screen)

            # Display score and game speed in the right corner
            score_text: Surface = settings.font.render(
                f"Score: {self.points}", True, (0, 0, 0)
            )
            self.screen.blit(score_text, (self.screen_width - 150, 20))

            speed_text: Surface = settings.font.render(
                f"Speed: {self.game_speed:.1f}", True, (0, 0, 0)
            )
            self.screen.blit(speed_text, (self.screen_width - 150, 50))

            # Display algorithm statistics in the left corner
            generation_text: Surface = settings.font.render(
                f"Generation: {n_generation}",
                True,
                (0, 0, 0),
            )
            self.screen.blit(generation_text, (20, 20))

            alive_text: Surface = settings.font.render(
                f"Alive: {sum(1 for dino in self.population_controller.population if dino.is_alive)}/{len(self.population_controller.population)}",
                True,
                (0, 0, 0),
            )
            self.screen.blit(alive_text, (20, 50))

            best_fitness_text: Surface = settings.font.render(
                f"Best Fitness: {self.population_controller.previous_best_fitness:.1f}",
                True,
                (0, 0, 0),
            )
            self.screen.blit(best_fitness_text, (20, 80))

            # Update score and game speed
            self.points += 1
            if self.points % 10 == 0:
                self.game_speed += settings.game_acceleration

            # Check if game over
            if not self.population_controller.check_population_alive():
                # Reinitialize population
                n_generation += 1
                self.population_controller.evolve_population()

                # Reset game state
                self.game_speed = settings.game_speed
                self.points = 0
                self.obstacles = []

            # Update display
            pygame.display.update()
            clock.tick(30)

    def _setup_display(self) -> None:
        """Sets up the display for the game"""
        pygame.display.set_caption("Dino Game")
        self.screen_height: int = settings.screen_height
        self.screen_width: int = settings.screen_width
        self.screen: Surface = pygame.display.set_mode(
            (self.screen_width, self.screen_height)
        )


def main() -> None:
    """Main function to run the game"""
    game = ChromeDinoGame()
    game.run()


if __name__ == "__main__":
    main()
