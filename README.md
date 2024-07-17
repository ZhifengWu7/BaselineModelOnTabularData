## Overview
Denoising diffusion models have led to significant advances in density modeling and image generation. In this work, we focus on denoising tabular data to further explore the feasibility of density estimation and tabular generation. Specifically, we use linear models, MLP models, and SVD to predict noise in synthetic tabular data under different signal-to-noise ratios, comparing these with the analytical solution of MMSE.

## Functions
- `noisy_channels(x, logsnr)`: Adds Guassian noise based on the SNR. Parameters include x for input data and logsnr for noise level.
- `train_and_evaluate(...)`: Trains a model on noisy data with L2 regularization and returns the validation loss. Implements early stopping to prevent overfitting. Key parameters include model, batch size, learning rate, and weight decay.
- `train_and_evaluate_l1(...)`: Trains a model on noisy data with L1 regularization and returns the validation loss. Implements early stopping to prevent overfitting. Key parameters include model, batch size, learning rate, and lambda for L1 regularization.
  
## Results
![image](https://github.com/ZhifengWu7/BaselineModelOnTabularData/assets/166958489/2dbc7714-6d38-4d43-af23-ad121043c2fe)
- The dashed line represents the MMSE Gaussian, which indicates the performance of the optimal denoiser. The baseline model can achieve the same performance as the optimal denoiser.
- The linear model performs well across various noise levels. At high noise levels, it nearly matches the optimal denoiser's performance, while at low noise levels, its performance drops slightly.
- The MLP model shows great performance, especially at high noise levels where it closely approaches or slightly surpasses the linear model. However, at low noise levels, it lags behind the linear model.
![image](https://github.com/ZhifengWu7/BaselineModelOnTabularData/assets/166958489/a05d19b7-64f6-4d3d-8d5f-48cd4d5514de)
- The plot shows the performance of models with L1 regularization. The linear model's performance has significantly improved, closely aligning with the optimal MSE.
