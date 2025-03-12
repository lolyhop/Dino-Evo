# Chrome Dino Game AI

A Python implementation of the Chrome Dinosaur Game with evolutionary algorithm that learns to play through population evolution.

## Overview

This project recreates the Chrome browser's dinosaur game and implements a neural network-based AI that learns to play the game through genetic evolution. The dinosaurs learn to jump over cacti and duck under birds to achieve higher scores.

## Repository structure

- `game.py`: Main game loop and rendering logic
- `dinosaur.py`: Dinosaur class with movement and neural network controller
- `entities.py`: Game entities
- `settings.py`: Game configuration and parameters
- `mlp.py`: Multi-layer perceptron neural network implementation for dinosaur decision making
- `population_controller.py`: Evolutionary algorithm for training dinosaurs
- `assets/`: Directory containing game sprites and images

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the game use the following command:
```bash
python game.py
```