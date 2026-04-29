# Tuning Deep Neural Networks on E-Commerce Data
Deep Learning Project · University of Tennessee, Knoxville

This project evaluates how deep neural network performance and runtime change under systematic hyperparameter tuning on e-commerce demand data.

## Project Overview
- Experimented with 16+ deep neural network configurations in TensorFlow to analyze sensitivity to tuning parameters and compare performance/runtime tradeoffs.
- Focused on practical model selection: balancing validation error, generalization, and training efficiency.

## Hyperparameter Grid
- **Network depth**: 2 to 4 hidden layers
- **Neuron widths**: 32 to 640 neurons
- **Activations**: ReLU, ELU, Leaky ReLU, Tanh, Sigmoid
- **Optimizers**: Adam, RMSprop, Adagrad, SGD with Nesterov momentum
- **Learning rates**: 0.0005 to 0.015
- **Batch sizes**: 32 to 256

## Best Model Configuration
- **Architecture**: 2 hidden layers (128 -> 64)
- **Activation**: ReLU
- **Optimizer**: Adam
- **Learning rate**: 0.001
- **Batch size**: 128
- **Training strategy**: Early stopping
- **Result**: Validation MAE of **12.54** in approximately **126 seconds**

## Key Findings
- Moderate complexity (2 to 3 layers) consistently outperformed deeper networks.
- Early stopping reduced overfitting and improved validation reliability.
- Embedding-style inputs (SKU + category) improved predictive quality.
- Regularization methods (dropout, batch normalization, L2) increased training stability.
- Smaller batch sizes and lower learning rates increased runtime substantially.

## Deliverables
- Results dashboard summarizing 16-experiment model grid outcomes
- Validation/test metrics comparison and runtime profiling
- Architecture summary and model-configuration documentation

## Tech Stack
Python · TensorFlow · VSCode · GitHub · Excel

## Skills Demonstrated
Python (Programming Language), TensorFlow, Deep Neural Networks (DNN), Mini-Batch Training, Feature Engineering, Pipeline Engineering, Machine Learning