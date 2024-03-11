# Simple XOR Neural Network Implementation

Welcome to the Simple XOR Neural Network Implementation repository! This Python project implements a simple artificial neural network (ANN) from scratch, trained to solve the classic XOR problem.

## Overview

The XOR problem involves mapping input values `(x1, x2)` to output values `(y)` such that the output is `1` if exactly one of the inputs is `1`, and `0` otherwise.

### Features

- **Implementation**: The neural network is built from scratch using Python and consists of a simple feedforward architecture with backpropagation for training.
  
- **Activation Function**: The network utilizes the sigmoid activation function and its derivative for computing gradients during backpropagation.
  
- **Training**: The network is trained iteratively by adjusting weights to minimize the error between predicted and actual output values.
  
- **Testing**: After training, the model is tested with all possible input pairs, and predictions are compared against the true output values to calculate accuracy.

## Usage

To use the provided code:

1. Clone this repository to your local machine.
2. Ensure you have Python 3.x installed.
3. Run the Python script `XOR_ANN.py`.
4. View the predictions and accuracy of the trained model.

## Requirements

- Python 3.x
- Required Python libraries: `math`, `random`

## Notes

This implementation serves as a foundational example of building and training a neural network in Python. For more complex problems or larger datasets, consider utilizing established deep learning frameworks like TensorFlow or PyTorch.
