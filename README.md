# Chrome Dino Game AI

A Python implementation of the Chrome Dinosaur Game with NEAT (NeuroEvolution of Augmenting Topologies) algorithm that learns to play through population evolution.

## Overview

This project recreates the Chrome browser's dinosaur game and implements a neural network-based AI that learns to play the game through genetic evolution. The dinosaurs learn to jump over cacti and duck under birds to achieve higher scores.

## Repository Structure

- `main.py`: Primary game application with visual interface showing dinosaurs learning in real-time
- `train.py`: Headless version of the game for faster training without rendering
- `settings.py`: Game configuration and parameters
- `requirements.txt`: Project dependencies

### Game Module
- `game/dinosaur.py`: Dinosaur class with movement and collision detection
- `game/dinosaur_controller.py`: Interface between the neural network and dinosaur actions
- `game/entities.py`: Game entities (obstacles, backgrounds, etc.)
- `game/population_controller.py`: Manages dinosaur population and evolution process

### NEAT Implementation
- `neat/ffn.py`: Feed-forward neural network implementation
- `neat/genome.py`: Genome class for storing network structure and weights
- `neat/evolutionary_operators.py`: Mutation and crossover operations
- `neat/activations.py`: Neural network activation functions
- `neat/edge.py`, `neat/node.py`, `neat/counter.py`: Supporting classes for network structure

### Utilities
- `utils/serialization.py`: Functions for saving and loading trained populations
- `utils/network_visualizer.py`: Tools for visualizing neural networks

### Other Directories
- `assets/`: Contains game sprites and images

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Playing with visual interface

To run the game with visual interface showing the dinosaurs learning in real-time:
```bash
python main.py
```

### Training without visualization

For faster training without graphical rendering:
```bash
python train.py
```

This headless mode saves training progress to TensorBoard logs and periodically saves the population to `population.json`.

### Monitoring training progress

To monitor training metrics with TensorBoard:
```bash
tensorboard --logdir=logs
```