from neat.ffn import FeedForwardNetwork
from neat.genome import Genome
from neat.activations import softmax
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
        """
        Predicts the action to be taken based on the provided features.

        This method takes in a numpy array of features, processes it through the
        FeedForwardNetwork, and returns the predicted action as a string. The action
        is selected based on the probabilities obtained from the softmax function
        applied to the network's output logits.

        Args:
            features (np.ndarray): A numpy array containing the input features for the network.

        Returns:
            str: The predicted action, which can be "nothing", "up", or "down".
        """

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
