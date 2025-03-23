from nn.genome import Genome
from typing import List


def generate_population(
    in_features: int, out_features: int, population_size: int = 10
) -> List[Genome]:
    return [Genome(in_features, out_features) for _ in range(population_size)]
