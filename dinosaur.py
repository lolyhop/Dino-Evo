from pygame import Rect, Surface
from pygame.image import load
from typing import Literal, Any
from dinosaur_controller import DinosaurController
from entities import Bird, Obstacle
from settings import settings
import pygame
import random
import numpy as np


class Dinosaur:
    """
    Represents the dinosaur character in the game.
    Controlled by a neural network.
    """

    def __init__(self, controller: DinosaurController) -> None:
        # Assets
        self.duck_img: list[Surface] = [
            load("assets/DinoDuck1.png"),
            load("assets/DinoDuck2.png"),
        ]
        self.run_img: list[Surface] = [
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
        self.color_mod: tuple[int, int, int] = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        self.transparency: int = 180

        # Dinosaur controller
        self.dino_controller: DinosaurController = controller
        self.fitness: int = 1

    def update(self, game_metadata: dict[str, float]) -> None:
        """
        Updates the state of the dinosaur based on the current game metadata.

        This method processes the game state to determine the appropriate action for the dinosaur,
        such as jumping or ducking. It also updates the dinosaur's position, animation state, and
        fitness score based on the actions taken.

        Args:
            game_metadata (dict[str, float]): A dictionary containing the current state of the game,
            which includes relevant information for decision making.

        Returns:
            None: This method modifies the internal state of the dinosaur in place.
        """
        if not self.is_alive:
            return None

        features: np.ndarray = self.extract_features(game_metadata)
        action: Literal["up", "down"] = self.dino_controller.predict_action(features)

        # Convert action to input
        userInput: dict[int, bool] = {pygame.K_UP: False, pygame.K_DOWN: False}
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

    def extract_features(self, game_metadata: dict[str, Any]) -> np.ndarray:
        """
        Extract features from the game state for AI decision making and convert to normalized numpy array.

        Args:
            game_metadata: dictionary containing game state information

        Returns:
            numpy array of normalized features for AI input
        """
        # Initialize default values
        distance_to_obstacle: int = settings.screen_width
        obstacle_type: str = "None"
        bird_height: int = 0
        obstacles: list[Obstacle] = game_metadata["obstacles"]
        game_speed: int = game_metadata["game_speed"]

        if obstacles:
            # Get the nearest obstacle
            next_obstacle: Obstacle = obstacles[0]
            distance_to_obstacle: int = next_obstacle.rect.x - self.dino_rect.x
            obstacle_type = type(next_obstacle).__name__

            # If the obstacle is a bird, get its height
            if isinstance(next_obstacle, Bird):
                bird_height: int = next_obstacle.rect.y

        # Extract numeric features
        feature_vector = np.array(
            [
                self.dino_rect.y,
                self.jump_vel,
                distance_to_obstacle,
                bird_height,
                game_speed,
            ]
        )

        # Encode obstacle type
        obstacle_encoding: list[int] = [
            0,
            0,
            0,
            0,
        ]  # [SmallCactus, LargeCactus, Bird, None]
        match obstacle_type:
            case "SmallCactus":
                obstacle_encoding[0] = 1
            case "LargeCactus":
                obstacle_encoding[1] = 1
            case "Bird":
                obstacle_encoding[2] = 1
            case "None":
                obstacle_encoding[3] = 1

        # Combine numeric features with obstacle encoding
        feature_vector = np.concatenate([feature_vector, obstacle_encoding])

        return feature_vector

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

        Returns:
            None: This method modifies the internal state of the dinosaur in place.
        """
        if not self.is_alive:
            return

        colored_image: Surface = self.image.copy()

        # Apply a color tint (ensure valid RGB values)
        r: int = max(0, min(255, self.color_mod[0]))
        g: int = max(0, min(255, self.color_mod[1]))
        b: int = max(0, min(255, self.color_mod[2]))
        color: tuple[int, int, int] = (r, g, b)
        colored_image.fill(color, special_flags=pygame.BLEND_RGB_MULT)

        # Apply transparency
        colored_image.set_alpha(self.transparency)

        screen.blit(colored_image, (self.dino_rect.x, self.dino_rect.y))
