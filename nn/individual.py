from nn.genome import Genome


class Individual:
    def __init__(self, genome: Genome, fitness: float):
        self.genome: Genome = genome
        self.fitness: float = fitness
