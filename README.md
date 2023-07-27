# Noedify

Noedify is a Neural Network library written in C# for running inference and limited live training in Unity 3D projects.

## Key Features

- Define a model architecture within a Unity project
- Save/load model files
- Import model parameters from Pytorch
- GPU accelerated inference
- Live training (linear and CNN layers only, CPU-multithreaded rather than GPU)


## Supported layer types:
- Linear/dense
- convolutional/transpose convolutional 2D
- convolutional/transpose convolutional 3D
- Pooling 2D
- 2D/3D batch normalization


# Example Projects

## Character Recognition

Draw on a canvas while a CNN classifier predicts the digit (0-9)