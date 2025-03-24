import json
from pathlib import Path

from neat.genome import Genome
from neat.node import Node
from neat.edge import Edge, Link
from neat.activations import relu, ACTIVATION_MAP, REVERSE_ACTIVATION_MAP


def serialize_population(population: list[Genome], path: str) -> None:
    """
    Serialize a population of Genome objects to a JSON file.

    Args:
        population: List of Genome objects to serialize
        path: Path to save the serialized data (file or directory)
    """
    # Prepare the path
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / "population.json"

    # Convert population to serializable format
    serialized_population = []

    for genome in population:
        serialized_genome = {
            "in_features": genome.in_features,
            "out_features": genome.out_features,
            "node_counter": genome.node_counter.value,
            "nodes": [],
            "edges": [],
        }

        # Serialize nodes
        for node in genome.nodes:
            activation_name = REVERSE_ACTIVATION_MAP.get(
                node.activation, "relu"
            )  # Default to relu if not found
            serialized_genome["nodes"].append(
                {
                    "id": node.id,
                    "bias": float(
                        node.bias
                    ),  # Convert numpy float to Python float if needed
                    "activation": activation_name,
                }
            )

        # Serialize edges
        for edge in genome.edges:
            serialized_genome["edges"].append(
                {
                    "input_id": edge.link.input_id,
                    "output_id": edge.link.output_id,
                    "weight": float(edge.weight),
                    "is_enabled": edge.is_enabled,
                }
            )

        serialized_population.append(serialized_genome)

    # Write to file
    with open(file_path, "w") as f:
        json.dump(serialized_population, f, indent=2)


def deserialize_population(path: str) -> list[Genome]:
    """
    Deserialize a population of Genome objects from a JSON file.

    Args:
        path: Path to load the serialized data from (file or directory)

    Returns:
        List of reconstructed Genome objects
    """
    # Prepare the path
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / "population.json"

    # Read from file
    with open(file_path, "r") as f:
        serialized_population = json.load(f)

    # Convert serialized data back to Genome objects
    population = []

    for serialized_genome in serialized_population:
        # Create a new Genome instance
        genome = Genome(
            serialized_genome["in_features"], serialized_genome["out_features"]
        )

        # Set the node counter
        genome.node_counter.value = serialized_genome["node_counter"]

        # Deserialize nodes
        for node_data in serialized_genome["nodes"]:
            activation_func = ACTIVATION_MAP.get(
                node_data["activation"], relu
            )  # Default to relu if not found
            node = Node(node_data["id"], node_data["bias"], activation_func)
            genome.add_node(node)

        # Deserialize edges
        for edge_data in serialized_genome["edges"]:
            link = Link(edge_data["input_id"], edge_data["output_id"])
            edge = Edge(link, edge_data["weight"], edge_data["is_enabled"])
            genome.add_edge(edge)

        population.append(genome)

    return population
