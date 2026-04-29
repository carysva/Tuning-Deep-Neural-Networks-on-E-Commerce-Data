# Incremental Learning on Large Data
Deep Learning Project · Feb 2026 - Feb 2026 · University of Tennessee, Knoxville

Excited to share my recent Deep Learning project: Incremental Learning on Large Data.

## Business Objective
Large training datasets often exceed available RAM on a typical laptop, which makes standard in-memory model training impractical. The objective of this project was to design and train a feed-forward neural network using an incremental mini-batch workflow that preserves model quality while staying memory-efficient.

## System Architecture (Memory-Efficient Training)
The pipeline was designed to stream data in chunks rather than loading the full dataset at once:

- **Data Source**: `pricing.csv` (large-scale e-commerce sales data)
- **Ingestion Strategy**: Chunked CSV processing with mini-batch loading
- **Feature Engineering**: `price`, `order`, `log(price)`, `log(order)`, and one-hot encoded `category`
- **Model Architecture**: 3 hidden layers (`96 -> 48 -> 24`) with sigmoid activation
- **Target Transform**: `log(1 + quantity)` to handle skew in quantity sold
- **Output Layer**: Linear output transformed back with `expm1` for quantity interpretation

## Key Results
- Built a feed-forward neural network in Python to predict product quantity sold on data too large to fit in RAM.
- Achieved `R^2 = 0.12` using incremental mini-batch learning.
- Completed training in approximately **143 seconds**.
- Maintained peak RAM usage at **299.6 MB**, demonstrating memory-efficient model design.
- Produced model diagnostics and interpretability assets:
  - Learning curve (incremental convergence behavior)
  - Permutation variable importance
  - Partial dependence plots for `price`, `order`, and `category`
- Variable importance analysis showed `category` and `price` as the dominant predictors, with `order` having lower relative influence.

## Interpretability and Diagnostics
- **Learning Curve**: Confirmed stable convergence as additional instances were learned.
- **Permutation Importance**: Quantified predictive contribution across core features.
- **Partial Dependence**:
  - `price`: Inverse relationship with predicted quantity.
  - `order`: Nonlinear trend with stronger effects at higher values.
  - `category`: Distinct category-level demand patterns.

## Technical Implementation Notes
- Implemented incremental training through mini-batch updates to avoid full-dataset memory loading.
- Standardized raw and log-transformed numeric features for stable optimization.
- Used one-hot encoding for categorical product grouping.
- Trained with Adam optimizer and MSE loss under constrained memory settings.
- Evaluated model quality with `R^2` and complementary interpretability diagnostics.

## Tech Stack
Python · TensorFlow · Keras · NumPy · Pandas · Matplotlib · VSCode · GitHub · Chunked CSV Processing · Mini-Batch Training · Permutation Importance · Partial Dependence Analysis

## Skills Demonstrated
Python (Programming Language), TensorFlow, Keras, NumPy, Pandas (software), Matplotlib, Chunked CSV Processing, Mini-Batch Training, Permutation Importance, Partial Dependence Analysis

## My Role
Model Developer · ML Pipeline Designer · Experiment Analyst

I designed the end-to-end incremental learning workflow, engineered memory-safe feature processing, implemented the neural network architecture, and produced the interpretability analysis used to explain model behavior and performance.