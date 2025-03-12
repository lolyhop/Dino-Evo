from __future__ import annotations

import random
from typing import List

from pygame import Rect, Surface
from pygame.image import load

from settings import settings


class Cloud:
    """
    Represents a cloud in the background.

    Clouds move from right to left and respawn when they go off-screen.
    """

    def __init__(self) -> None:
        """Initialize a cloud with random position."""
        self.x: int = settings.screen_width + random.randint(900, 1000)
        self.y: int = random.randint(65, 100)
        self.image: Surface = load("assets/Cloud.png")
        self.width: int = self.image.get_width()

    def update(self, game_speed: int) -> None:
        """Update cloud position and respawn if off-screen."""
        self.x -= game_speed
        if self.x < -self.width:
            self.x = settings.screen_width + random.randint(2500, 3000)
            self.y = random.randint(65, 100)

    def draw(self, screen: Surface) -> None:
        """
        Draw the cloud on the screen.

        Args:
            screen: Pygame surface to draw on
        """
        screen.blit(self.image, (self.x, self.y))


class Obstacle:
    """
    Base class for obstacles in the game.

    Obstacles move from right to left and are removed when they go off-screen.
    """

    def __init__(self, assets: List[Surface], type_idx: int) -> None:
        """
        Initialize an obstacle.

        Args:
            assets: List of possible obstacle images
            type_idx: Index of the image to use
        """
        self.image: List[Surface] = assets
        self.type: int = type_idx
        self.rect: Rect = self.image[self.type].get_rect()
        self.rect.x = settings.screen_width

    def update(self, game_speed: int) -> None:
        """Update obstacle position and remove if off-screen."""
        self.rect.x -= game_speed

    def draw(self, screen: Surface) -> None:
        """
        Draw the obstacle on the screen.

        Args:
            screen: Pygame surface to draw on
        """
        screen.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    """Small cactus obstacle."""

    def __init__(self) -> None:
        self.type: int = random.randint(0, 2)
        self.small_cactus_assets: List[Surface] = [
            load("assets/SmallCactus1.png"),
            load("assets/SmallCactus2.png"),
            load("assets/SmallCactus3.png"),
        ]
        super().__init__(self.small_cactus_assets, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    """Large cactus obstacle."""

    def __init__(self) -> None:
        self.type: int = random.randint(0, 2)
        self.large_cactus_assets: List[Surface] = [
            load("assets/LargeCactus1.png"),
            load("assets/LargeCactus2.png"),
            load("assets/LargeCactus3.png"),
        ]
        super().__init__(self.large_cactus_assets, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    """
    Flying bird obstacle with animation.

    Birds are positioned higher than cacti and have wing flapping animation.
    """

    def __init__(self) -> None:
        self.type: int = 0
        self.index: int = 0
        self.bird_assets: List[Surface] = [
            load("assets/Bird1.png"),
            load("assets/Bird2.png"),
        ]
        super().__init__(self.bird_assets, self.type)
        self.rect.y = 250

    def draw(self, screen: Surface) -> None:
        """
        Draw the bird with wing flapping animation.

        Args:
            screen: Pygame surface to draw on
        """
        if self.index >= 10:
            self.index = 0
        screen.blit(self.image[self.index // 5], self.rect)
        self.index += 1


class Background:
    """
    Represents the background of the game.
    """

    def __init__(self) -> None:
        self.background: Surface = load("assets/Track.png")
        self.image_width: int = self.background.get_width()
        self.x_pos_bg: int = 0
        self.y_pos_bg: int = 380
        self.game_speed: int = settings.game_speed

    def update(self, screen: Surface, game_speed: int) -> None:
        """
        Update and draw the scrolling background.

        Args:
            screen: Pygame surface to draw on
        """

        screen.blit(self.background, (self.x_pos_bg, self.y_pos_bg))
        screen.blit(self.background, (self.image_width + self.x_pos_bg, self.y_pos_bg))

        if self.x_pos_bg <= -self.image_width:
            screen.blit(
                self.background, (self.image_width + self.x_pos_bg, self.y_pos_bg)
            )
            self.x_pos_bg = 0

        self.x_pos_bg -= game_speed

    def draw(self, screen: Surface) -> None:
        """
        Draw the background on the screen.
        """
        screen.blit(self.background, (self.x_pos_bg, self.y_pos_bg))
        screen.blit(self.background, (self.image_width + self.x_pos_bg, self.y_pos_bg))
