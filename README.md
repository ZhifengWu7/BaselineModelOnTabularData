# Noise Modeling and Estimation

This Python code uses PyTorch, Matplotlib, and Scikit-Learn to model and estimate noise in synthetic tabular data using linear and multilayer perceptron (MLP) models.

## Requirements
- Python 3.8+
- PyTorch
- Matplotlib
- Scikit-Learn

## Overview
The script generates synthetic data and applies Gaussian noise to simulate different signal-to-noise ratios (SNRs). It includes two model architectures:
- `LinearRegressionModel`: A simple linear regression model.
- `MLPModel`: A multilayer perceptron model with optional dropout.

The models are trained to predict the noise added to the synthetic data, and their performance is evaluated across a range of logarithmic SNR values. The results are plotted to compare the mean squared error (MSE) of each model against the baseline and minimum MSE (MMSE).

## Usage
To run the script, ensure all dependencies are installed and execute the Python file:
```bash
python noise_modeling.py
