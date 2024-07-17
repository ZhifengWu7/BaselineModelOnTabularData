import torch as t
import matplotlib.pyplot as plt
from data_generation import generate_tabular_data
from models import LinearRegressionModel, MLPModel
from train_eval import train_and_evaluate, compute_baseline_mse
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
svd_mses = []
mmse_g = []

for logsnr in logsnrs:
    linear_model = LinearRegressionModel(in_dim=rows*columns, out_dim=(rows,columns))
    mlp_model = MLPModel(in_dim=rows*columns, out_dim=(rows,columns), dropout=dropout)
    val_loss_linear = train_and_evaluate(x=x, logsnr=logsnr, model=linear_model, batch_size=batch_size, num_epochs=num_epochs_linear, learning_rate=learning_rate, weight_decay=weight_decay)
    val_loss_mlp = train_and_evaluate(x=x, logsnr=logsnr, model=mlp_model, batch_size=batch_size, num_epochs=num_epochs_mlp, learning_rate=learning_rate, weight_decay=weight_decay)
    linear_mses.append(val_loss_linear
