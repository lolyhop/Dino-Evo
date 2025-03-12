from typing import Optional

from pydantic_settings import BaseSettings
from pygame.font import Font


class Settings(BaseSettings):
    # Game settings
    screen_width: int = 1024
    screen_height: int = 800
    game_speed: int = 20
    max_game_speed: int = 50
    game_acceleration: float = 0.1
    font: Optional[Font] = None  # Will be initialized after pygame.init() is called

    # NEAT settings
    population_size: int = 20
    selection_amount: float = 0.5  # percentage of population to select
    mutation_rate: float = 0.2  # probability of mutation
    mutation_scale: float = 0.1  # scale of mutation
    stagnation_threshold: float = (
        0.1  # if we have not improved by 10% in the last generation, we are stagnating
    )
    stagnation_replacement_percentage: float = (
        0.2  # percentage of population to replace with new dinosaurs
    )

    def initialize_font(self):
        """Initialize the font after pygame is initialized."""
        self.font = Font("freesansbold.ttf", 20)


settings = Settings()
