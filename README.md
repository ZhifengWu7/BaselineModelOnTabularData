## Overview
The script generates synthetic tabular data and applies Gaussian noise to simulate different signal-to-noise ratios (SNRs). It includes two model architectures:
- `LinearRegressionModel`: A simple linear regression model.
- `MLPModel`: A multilayer perceptron model with a residual connection and optional dropout.

The models are trained to predict the noise added to the synthetic data, and their performance is evaluated across a range of logarithmic SNR values. The results are plotted to compare the mean squared error (MSE) of each model against the baseline and minimum MSE (MMSE). The baseline model is defined by the formula eps_hat = t.sqrt(t.sigmoid(-logsnr)) * z.

## Functions
- `noisy_channels(x, logsnr)`: Adds Guassian noise based on the SNR. Parameters include x for input data and logsnr for noise level.
- `train_and_evaluate(...)`: Trains a model on noisy data with L2 regularization and returns the validation loss. Implements early stopping to prevent overfitting. Key parameters include model, batch size, learning rate, and weight decay.
- train_and_evaluate_l1(...): Trains a model on noisy data with L1 regularization and returns the validation loss. Implements early stopping to prevent overfitting. Key parameters include model, batch size, learning rate, and lambda for L1 regularization.
  
## Results
![image](https://github.com/ZhifengWu7/BaselineModelOnTabularData/assets/166958489/332994b0-9d15-4bd4-a27c-7c1987792888)
- The dashed line represents the MMSE Gaussian, which indicates the performance of the optimal denoiser. The baseline model can achieve the same performance as the optimal denoiser.
- The linear model performs well across various noise levels. At high noise levels, it nearly matches the optimal denoiser's performance, while at low noise levels, its performance drops slightly.
- The MLP model shows great performance, especially at high noise levels where it closely approaches or slightly surpasses the linear model. However, at low noise levels, it lags behind the linear model. Notably, the MLP model achieves its best performance in fewer epochs compared to the linear model.
