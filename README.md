## Overview
Denoising diffusion models have led to significant advances in density modeling and image generation. In this work, we focus on denoising tabular data to further explore the feasibility of density estimation and tabular generation. Specifically, we use linear models, MLP models, and SVD to predict noise in synthetic tabular data under different signal-to-noise ratios, comparing these with the analytical solution of MMSE.

## Functions
- `noisy_channels(x, logsnr)`: Adds Guassian noise based on the SNR. Parameters include x for input data and logsnr for noise level.
- `train_and_evaluate(...)`: Trains a model on noisy data with L2 regularization and returns the validation loss. Implements early stopping to prevent overfitting. Key parameters include model, batch size, learning rate, and weight decay.
- `train_and_evaluate_l1(...)`: Trains a model on noisy data with L1 regularization and returns the validation loss. Implements early stopping to prevent overfitting. Key parameters include model, batch size, learning rate, and lambda for L1 regularization.
  
## Results

![image](https://github.com/user-attachments/assets/26c944a0-b1b8-47d5-a72e-a0597ef98774)

