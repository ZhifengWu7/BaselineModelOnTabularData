import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from data_generation import generate_tabular_data
from models import LinearRegressionModel, MLPModel
from train_eval import train_and_evaluate, compute_baseline_mse, noisy_channels
from svd_eval import svd_predict_and_evaluate

# Define parameters
n_samples = 50000
rows = 30
columns = 20
rank = 1
tail_strength = 0.1
random_state = 42
batch_size = 50
num_epochs_linear = 1
num_epochs_mlp = 1
learning_rate = 0.001
dropout = 0
weight_decay = 0.1
loc, s = 0, 1
logsnrs = t.linspace(loc-4*s, loc+4*s, 10)

# Generate data
x = generate_tabular_data(n_samples, rows, columns, rank, tail_strength, random_state)

# Train and evaluate models
linear_mses = []
mlp_mses = []
baseline_mses = []
svd_mses = []
mmse_g = []

# Train and evaluate models for each logsnr value
for logsnr in logsnrs:
    linear_model = LinearRegressionModel(in_dim=rows*columns, out_dim=(rows,columns))
    mlp_model = MLPModel(in_dim=rows*columns, out_dim=(rows,columns), dropout=dropout)
    val_loss_linear = train_and_evaluate(x=x, logsnr=logsnr, model=linear_model, batch_size=batch_size, num_epochs=num_epochs_linear, learning_rate=learning_rate, weight_decay=weight_decay)
    val_loss_mlp = train_and_evaluate(x=x, logsnr=logsnr, model=mlp_model, batch_size=batch_size, num_epochs=num_epochs_mlp, learning_rate=learning_rate, weight_decay=weight_decay)
    linear_mses.append(val_loss_linear)
    mlp_mses.append(val_loss_mlp)

for logsnr in logsnrs:
    val_loss_svd = svd_predict_and_evaluate(x=x, logsnr=logsnr, batch_size=batch_size, rank=rank)
    svd_mses.append(val_loss_svd)

# Compute baseline MSE
criterion = nn.MSELoss(reduction='sum')
for logsnr in logsnrs:
    logsnr = t.ones(len(x)) * logsnr
    z, eps = noisy_channels(x, logsnr)
    dims = tuple(1 for _ in range(len(x[0].shape)))
    left = (-1,) + dims
    logsnr = logsnr.view(left)

    eps_hat = t.sqrt(t.sigmoid(-logsnr)) * z

    baseline_mse = criterion(eps_hat, eps).item() / len(x)
    baseline_mses.append(baseline_mse)

mmse_g = []
mmse_g = rows * columns * t.sigmoid(logsnrs)

plt.figure(figsize=(10, 6))
plt.plot(logsnrs.numpy(), svd_mses, label='SVD Decomposition')
plt.plot(logsnrs.numpy(), linear_mses, label='Linear Model')
plt.plot(logsnrs.numpy(), mlp_mses, label='MLP Model')
plt.plot(logsnrs.numpy(), baseline_mses, label='Baseline MSE')
plt.plot(logsnrs.numpy(), mmse_g, label='MMSE', linestyle='--')
plt.xlabel('Log SNR')
plt.ylabel('MSE')
plt.title('MSE vs. Log SNR for Multiple Models')
plt.legend()
plt.grid(True)
plt.show()
