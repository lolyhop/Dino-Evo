from __future__ import annotations
from typing import Dict, Any, List, Optional

import numpy as np


class MLP:
    """
    Multi-Layer Perceptron (MLP) neural network.
    """

    def __init__(self, weights: Optional[Dict[str, np.ndarray]] = None) -> None:
        # TODO: For now we have hardcoded MLP architecture
        # However, since we use Genetic Programming, we have to vary the architecture
        # to find the best one for the task
        self.input_size: int = 9
        self.hidden1_size: int = 20
        self.hidden2_size: int = 20
        self.output_size: int = 3  # nothing, up, down

        # Initialize weights randomly if not provided
        if weights is None:
            # Initialize weights with Xavier initialization
            self.weights = {
                "W1": np.random.randn(self.input_size, self.hidden1_size)
                * np.sqrt(2 / self.input_size),
                "b1": np.zeros((1, self.hidden1_size)),
                "W2": np.random.randn(self.hidden1_size, self.hidden2_size)
                * np.sqrt(2 / self.hidden1_size),
                "b2": np.zeros((1, self.hidden2_size)),
                "W3": np.random.randn(self.hidden2_size, self.output_size)
                * np.sqrt(2 / self.hidden2_size),
                "b3": np.zeros((1, self.output_size)),
            }
        else:
            self.weights = weights

        # Define action mapping
        self.actions = {0: "nothing", 1: "up", 2: "down"}

    def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dictionary to normalized numpy array."""
        # Extract numeric features
        feature_vector = np.array(
            [
                features["dino_y"],
                features["dino_jump_vel"],
                features["distance_to_obstacle"],
                features["bird_height"],
                features["obstacle_velocity"],
            ]
        ).reshape(1, -1)

        obstacle_type: str = features["obstacle_type"]
        obstacle_encoding: List[int] = [0, 0, 0, 0]  # [SmallCactus, LargeCactus, Bird, None]
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
        feature_vector = np.hstack(
            [feature_vector, np.array(obstacle_encoding).reshape(1, -1)]
        )

        # TODO: Normalize features
        return feature_vector

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Perform forward pass through the network."""
        # Hidden layer 1
        z1 = np.dot(X, self.weights["W1"]) + self.weights["b1"]
        a1 = self._relu(z1)

        # Hidden layer 2
        z2 = np.dot(a1, self.weights["W2"]) + self.weights["b2"]
        a2 = self._relu(z2)

        # Output layer
        z3 = np.dot(a2, self.weights["W3"]) + self.weights["b3"]

        return z3

    def predict_action(self, features: Dict[str, Any]) -> str:
        """Predict the next action based on current game state."""
        # Preprocess features
        X = self._preprocess_features(features)

        # Forward pass
        output = self._forward_pass(X)

        # Get action with highest score
        # TODO: To sample from softmax probs (?)
        prediction = np.argmax(output)
        return self.actions[prediction]

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return the current weights of the network."""
        # TODO: Rewrite as pythonic getter
        return self.weights

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set new weights for the network."""
        # TODO: Rewrite as pythonic setter
        self.weights = weights

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1) -> None:
        """
        Mutate the weights of the network.

        1. Creates a boolean mask where True indicates weights to be mutated
        2. Generates random values from a normal distribution scaled by mutation_scale
        3. Adds the scaled random values to the selected weights

        Args:
            mutation_rate: Float between 0 and 1 determining the probability of
                           each weight being mutated (default: 0.1)
            mutation_scale: Float determining the magnitude of mutations (default: 0.1)
        """
        for key in self.weights:
            # Create mutation mask (True/False) based on mutation rate
            mutation_mask = np.random.random(self.weights[key].shape) < mutation_rate

            # Create random mutations
            mutations = np.random.randn(*self.weights[key].shape) * mutation_scale

            # Apply mutations only where mask is True
            self.weights[key] = self.weights[key] + mutations * mutation_mask

    def crossover(self, mlp: MLP) -> MLP:
        """
        Perform a uniform crossover with another MLP.

        1. Get weights for each MLP layer
        2. For each weight matrix:
           - Create a uniform random binary mask
           - Take weights from self where mask is True
           - Take weights from other MLP where mask is False
        3. Return a new MLP with the combined weights

        Args:
            mlp: Another MLP instance to crossover with

        Returns:
            A new MLP instance with weights inherited from both parents
        """
        other_weights: Dict[str, np.ndarray] = mlp.get_weights()
        child_weights: Dict[str, np.ndarray] = {}

        for key in self.weights:
            # Create random mask for crossover
            mask: np.ndarray = np.random.random(self.weights[key].shape) > 0.5

            # Take weights from self where mask is True, from other where mask is False
            child_weights[key] = np.where(mask, self.weights[key], other_weights[key])

        return MLP(weights=child_weights)
