import random
from typing import List, Dict, Any, Literal, Tuple

import pygame
from pygame import Rect, Surface
from pygame.image import load

from mlp import MLP
from entities import Bird, Obstacle
from settings import settings


class Dinosaur:
    """
    Represents the dinosaur character in the game.
    Controlled by a neural network.
    """

    def __init__(self, controller: MLP) -> None:
        # Assets
        self.duck_img: List[Surface] = [
            load("assets/DinoDuck1.png"),
            load("assets/DinoDuck2.png"),
        ]
        self.run_img: List[Surface] = [
            load("assets/DinoRun1.png"),
            load("assets/DinoRun2.png"),
        ]
        self.jump_img: Surface = load("assets/DinoJump.png")

        # Movement states
        self.dino_duck: bool = False
        self.dino_run: bool = True
        self.dino_jump: bool = False
        self.is_alive: bool = True

        # Animation and physics
        self.x_pos: int = 100
        self.y_pos: int = 310
        self.y_pos_duck: int = 340
        self.jump_velocity: float = 8.5
        self.jump_vel: float = self.jump_velocity
        self.step_index: int = 0
        self.image: Surface = self.run_img[0]
        self.dino_rect: Rect = self.image.get_rect()
        self.dino_rect.x = self.x_pos
        self.dino_rect.y = self.y_pos
        self.color_mod: Tuple[int, int, int] = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        self.transparency: int = 180

        # Dinosaur controller
        self.dino_controller: MLP = controller
        self.fitness: int = 1

    def update(self, game_metadata: Dict[str, Any]) -> None:
        """
        Update the dinosaur's state based on AI decisions or user input.

        Args:
            obstacles: List of obstacles in the game
        """
        if not self.is_alive:
            return None

        features: Dict[str, Any] = self.extract_features(game_metadata)
        action: Literal["up", "down"] = self.dino_controller.predict_action(features)

        # Convert action to input
        userInput: Dict[int, bool] = {pygame.K_UP: False, pygame.K_DOWN: False}
        if action == "up":
            userInput[pygame.K_UP] = True
        elif action == "down":
            userInput[pygame.K_DOWN] = True

        # Update movement state
        if userInput[pygame.K_UP] and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

        # Update animation and position based on movement state
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        # Update fitness
        self.fitness += 1

    def extract_features(self, game_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from the game state for AI decision making.

        Args:
            obstacles: List of obstacles in the game

        Returns:
            Dictionary of features for AI input
        """
        # Initialize default values
        distance_to_obstacle: int = settings.screen_width
        obstacle_type: str = "None"
        bird_height: int = 0
        obstacles: List[Obstacle] = game_metadata["obstacles"]
        game_speed: int = game_metadata["game_speed"]

        if obstacles:
            # Get the nearest obstacle
            next_obstacle: Obstacle = obstacles[0]
            distance_to_obstacle: int = next_obstacle.rect.x - self.dino_rect.x
            obstacle_type = type(next_obstacle).__name__

            # If the obstacle is a bird, get its height
            if isinstance(next_obstacle, Bird):
                bird_height: int = next_obstacle.rect.y

        # Create a feature vector
        features = {
            "dino_y": self.dino_rect.y,
            "dino_jump_vel": self.jump_vel,
            "distance_to_obstacle": distance_to_obstacle,
            "obstacle_type": obstacle_type,
            "bird_height": bird_height,
            "obstacle_velocity": game_speed,
        }
        return features

    def duck(self) -> None:
        """Lower dinosaur position and update animation for ducking."""
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.x_pos
        self.dino_rect.y = self.y_pos_duck
        self.step_index += 1

    def run(self) -> None:
        """Update dinosaur animation for running."""
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.x_pos
        self.dino_rect.y = self.y_pos
        self.step_index += 1

    def jump(self) -> None:
        """Update dinosaur position for jumping with gravity effect."""
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8  # gravity
        if self.jump_vel < -self.jump_velocity:
            self.dino_jump = False
            self.jump_vel = self.jump_velocity

    def draw(self, screen: Surface) -> None:
        """
        Draw the dinosaur on the screen with optional color modification.

        Args:
            screen: Pygame surface to draw on
        """
        if not self.is_alive:
            return

        colored_image: Surface = self.image.copy()

        # Apply a color tint (ensure valid RGB values)
        r: int = max(0, min(255, self.color_mod[0]))
        g: int = max(0, min(255, self.color_mod[1]))
        b: int = max(0, min(255, self.color_mod[2]))
        color: Tuple[int, int, int] = (r, g, b)
        colored_image.fill(color, special_flags=pygame.BLEND_RGB_MULT)

        # Apply transparency
        colored_image.set_alpha(self.transparency)

        screen.blit(colored_image, (self.dino_rect.x, self.dino_rect.y))
