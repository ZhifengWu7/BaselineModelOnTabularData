Noise Prediction Models on Tabular Data

Overview  
This repository contains implementation of linear and MLP (Multi-Layer Perceptron) models designed to predict noise levels at various log SNR (Signal-to-Noise Ratio) levels. These models calculate the Mean Squared Error (MSE) of noise predictions to evaluate performance.

Models  
Linear Model: A straightforward approach using linear regression techniques to predict the noise level from tabular data based on log SNR values.
MLP Model: An advanced model utilizing a multi-layer perceptron that learns complex patterns in the data to make more accurate predictions compared to the linear model.

Baseline Comparison  
To assess the effectiveness of our predictive models, we also plot the MMSE (Minimum Mean Square Error) Gaussian. This serves as a benchmark to compare how close the MSE from our models approaches the theoretical minimum error.

Results  
The results section will include plots that demonstrate the MSE achieved by both the linear and MLP models across various SNR levels, compared against the MMSE Gaussian curve.
![image](https://github.com/ZhifengWu7/BaselineModelOnTabularData/assets/166958489/14d1e235-6a73-4f3e-a547-d52fe2899866)

