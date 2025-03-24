import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from nn.genome import Genome


def create_network_image(genome: Genome, filename: str = None):
    """
    Create an image of the neural network and optionally save it to a file.

    Args:
        genome: The genome to visualize
        filename: If provided, save the image to this file path

    Returns:
        The matplotlib figure object
    """
    visualizer = NetworkVisualizer(genome)
    visualizer._calculate_node_positions()
    visualizer._draw_edges()
    visualizer._draw_nodes()
    visualizer._add_labels()

    visualizer.ax.set_xlim(-0.1, 1.1)
    visualizer.ax.set_ylim(-0.1, 1.1)
    visualizer.ax.axis("off")
    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    return visualizer.fig


class NetworkVisualizer:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.node_positions = {}
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def visualize(self):
        """Visualize the neural network defined by the genome."""
        self._calculate_node_positions()
        self._draw_edges()
        self._draw_nodes()
        self._add_labels()

        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.axis("off")
        plt.tight_layout()
        plt.show()

    def _calculate_node_positions(self):
        """Calculate positions for all nodes in the network."""
        # Get node categories
        input_nodes = self.genome.get_input_or_hidden_nodes()
        output_nodes = self.genome.get_output_nodes()
        hidden_nodes = self.genome.get_hidden_nodes()

        # Remove hidden nodes from input nodes list
        input_nodes = [
            node_id for node_id in input_nodes if node_id not in hidden_nodes
        ]

        # Calculate positions
        self._position_input_nodes(input_nodes)
        self._position_hidden_nodes(hidden_nodes)
        self._position_output_nodes(output_nodes)

    def _position_input_nodes(self, input_nodes: list[int]):
        """Position input nodes at the left side of the visualization."""
        num_nodes = len(input_nodes)
        if num_nodes == 0:
            return

        y_step = 0.8 / max(1, num_nodes - 1) if num_nodes > 1 else 0
        for i, node_id in enumerate(input_nodes):
            self.node_positions[node_id] = (0.1, 0.1 + i * y_step)

    def _position_output_nodes(self, output_nodes: list[int]):
        """Position output nodes at the right side of the visualization."""
        num_nodes = len(output_nodes)
        if num_nodes == 0:
            return

        y_step = 0.8 / max(1, num_nodes - 1) if num_nodes > 1 else 0
        for i, node_id in enumerate(output_nodes):
            self.node_positions[node_id] = (0.9, 0.1 + i * y_step)

    def _position_hidden_nodes(self, hidden_nodes: list[int]):
        """Position hidden nodes in the middle of the visualization."""
        num_nodes = len(hidden_nodes)
        if num_nodes == 0:
            return

        # Arrange in a grid if there are many hidden nodes
        cols = max(1, int(np.sqrt(num_nodes)))
        rows = (num_nodes + cols - 1) // cols

        x_step = 0.5 / max(1, cols)
        y_step = 0.8 / max(1, rows)

        for i, node_id in enumerate(hidden_nodes):
            col = i % cols
            row = i // cols
            # Move hidden nodes further to the right (from 0.2 to 0.4)
            x = 0.4 + col * x_step
            y = 0.1 + row * y_step
            self.node_positions[node_id] = (x, y)

    def _draw_nodes(self):
        """Draw all nodes in the network."""
        input_nodes = self.genome.get_input_or_hidden_nodes()
        output_nodes = self.genome.get_output_nodes()
        hidden_nodes = self.genome.get_hidden_nodes()

        # Remove hidden nodes from input nodes list
        input_nodes = [
            node_id for node_id in input_nodes if node_id not in hidden_nodes
        ]

        # Draw each type of node with different colors
        for node_id in input_nodes:
            self._draw_node(node_id, "lightblue")

        for node_id in hidden_nodes:
            self._draw_node(node_id, "lightgreen")

        for node_id in output_nodes:
            self._draw_node(node_id, "salmon")

    def _draw_node(self, node_id: int, color: str):
        """Draw a single node as a circle with its ID inside."""
        if node_id not in self.node_positions:
            return

        x, y = self.node_positions[node_id]
        # Draw a slightly larger white circle first to create a background
        bg_circle = plt.Circle((x, y), 0.055, fill=True, color="white", zorder=10)
        self.ax.add_patch(bg_circle)
        # Draw the colored circle on top
        circle = plt.Circle((x, y), 0.05, fill=True, color=color, alpha=0.8, zorder=20)
        self.ax.add_patch(circle)
        # Draw the text with highest z-order to ensure it's on top
        self.ax.text(
            x, y, str(node_id), ha="center", va="center", fontsize=9, zorder=30
        )

    def _draw_edges(self):
        """Draw all edges in the network."""
        for edge in self.genome.edges:
            if not edge.is_enabled:
                continue

            input_id = edge.link.input_id
            output_id = edge.link.output_id

            if (
                input_id not in self.node_positions
                or output_id not in self.node_positions
            ):
                continue

            start_x, start_y = self.node_positions[input_id]
            end_x, end_y = self.node_positions[output_id]

            # Draw the edge with weight-based thickness
            weight = abs(edge.weight)
            color = "blue" if edge.weight >= 0 else "red"
            linewidth = 0.5 + weight * 2

            # Set a lower zorder to ensure edges are drawn behind nodes
            self.ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                color=color,
                linewidth=linewidth,
                alpha=0.6,
                zorder=1,
            )

    def _add_labels(self):
        """Add a legend to explain node and edge colors."""
        input_patch = patches.Patch(color="lightblue", label="Input Nodes")
        hidden_patch = patches.Patch(color="lightgreen", label="Hidden Nodes")
        output_patch = patches.Patch(color="salmon", label="Output Nodes")
        pos_edge = plt.Line2D(
            [0], [0], color="blue", linewidth=2, label="Positive Weight"
        )
        neg_edge = plt.Line2D(
            [0], [0], color="red", linewidth=2, label="Negative Weight"
        )

        self.ax.legend(
            handles=[input_patch, hidden_patch, output_patch, pos_edge, neg_edge],
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=3,
        )


def visualize_network(genome: Genome):
    """Convenience function to visualize a genome."""
    visualizer = NetworkVisualizer(genome)
    visualizer.visualize()
