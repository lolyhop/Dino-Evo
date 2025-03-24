from nn.ffn import FeedForwardNetwork
from nn.genome import Genome
from nn.activations import softmax
import numpy as np


class DinosaurController:
    """
    Multi-Layer Perceptron (MLP) neural network.
    """

    def __init__(
        self,
        genome: Genome,
    ) -> None:
        # Define action mapping
        self.genome: Genome = genome
        self.actions = ["nothing", "up", "down"]

    def predict_action(self, features: np.ndarray) -> str:
        """Predict the next action based on current game state."""

        ffn: FeedForwardNetwork = FeedForwardNetwork(self.genome)

        # Forward pass
        try:
            logits = ffn.forward(features)
            probabilities = softmax(logits)

            prediction = np.random.choice(
                self.actions, p=probabilities, size=1, replace=False
            )[0]
        except Exception as e:
            print(f"Error predicting action: {e}")
            prediction = "nothing"

        return prediction
