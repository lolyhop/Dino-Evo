from nn.genome import Genome
from typing import List


def generate_population(
    in_features: int, out_features: int, population_size: int = 10
) -> List[Genome]:
    genomes: List[Genome] = []
    for _ in range(population_size):
        genome: Genome = Genome(in_features, out_features)
        genome.initialize_genome()
        genomes.append(genome)
    return genomes
