## Overview
Denoising diffusion models have led to significant advances in density modeling and image generation. In this work, we focus on **denoising tabular data** to further explore the feasibility of density estimation and tabular generation. Specifically, I use linear models, MLP models, and SVD to predict noise in synthetic tabular data under different signal-to-noise ratios, comparing these with the analytical solution of MMSE.

## Requirements

You can install requirements using pip:

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/ZhifengWu7/BaselineModelOnTabularData.git
cd BaselineModelOnTabularData
```
2. Run the main script:
```bash
python main.py
```

## Results
The SVD approach yielded the best results, followed by the baseline MSE. After tuning, the MLP model performed better than the tuned linear model but was still slightly worse than the baseline MSE. 

Note that we assumed that the rank of the synthetic tabular data was a known parameter when running SVD.
![image](https://github.com/user-attachments/assets/26c944a0-b1b8-47d5-a72e-a0597ef98774)
