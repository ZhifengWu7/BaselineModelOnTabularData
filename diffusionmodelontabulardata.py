import torch as t
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.linalg
from sklearn.datasets import make_low_rank_matrix
from torch.utils.data import TensorDataset, DataLoader, random_split

# Generate tabular data
n_samples = 50000  # Number of samples
rows = 30  # Number of rows in each matrix
columns = 20  # Number of columns in each matrix
rank = 5  # the rank of the matrix
tail_strength = 0.1
random_state = 42

# Generate tabular data
x = np.empty((n_samples, rows, columns))
for i in range(n_samples):
    x[i] = make_low_rank_matrix(n_samples=rows, n_features=columns, effective_rank=rank, tail_strength=tail_strength, random_state=random_state+i)

# Normalize the data by subtracting the mean and dividing by the standard deviation.
x_np = np.array(x)
means = np.mean(x_np, axis=0)
std_devs = np.std(x_np, axis=0)
x_normalized = (x_np - means) / std_devs

x = t.tensor(x_normalized, dtype=t.float32)

def noisy_channels(x, logsnr):
    """Add Gaussian noise to x, return "z" and epsilon."""
    dims = tuple(1 for _ in range(len(x[0].shape)))
    left = (-1,) + dims
    logsnr = logsnr.view(left)
    eps = t.randn((len(logsnr),)+x[0].shape)
    return t.sqrt(t.sigmoid(logsnr))*x + t.sqrt(t.sigmoid(-logsnr))*eps, eps

class LinearRegressionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegressionModel, self).__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # Reshape the input tensor 'x' to a 2D tensor with shape (batch_size, -1)
        x = x.view(x.size(0), -1)
        # Apply the linear transformation to the input
        x = self.linear(x)
        # Reshape the output tensor to the desired output dimensions
        return x.view(x.size(0), *self.out_dim)

class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c_fc = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(in_dim, in_dim)
        # Define a dropout layer if dropout rate is greater than 0, else set it to None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # Reshape the input tensor 'x' to a 2D tensor with shape (batch_size, -1)
        x = x.view(x.size(0), -1)
        # Store the input tensor in 'residual' for the residual connection
        residual = x
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        # Apply dropout if it is defined
        if self.dropout:
          x = self.dropout(x)
        # Add the residual (input tensor) to the output tensor (residual connection)
        x += residual
        # Reshape the output tensor to the desired output dimensions
        return x.view(x.size(0), *self.out_dim)

def train_and_evaluate(x, logsnr, model, batch_size, num_epochs, learning_rate, weight_decay):
    """Given a specific logsnr value, train the model and return val_loss."""
    dataset = TensorDataset(x)
    n = len(dataset)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * n)
    val_size = n - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Setup the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='sum')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch_x = batch[0]
            # Add noise to the input data
            z, eps = noisy_channels(batch_x, t.ones(len(batch_x)) * logsnr)
            optimizer.zero_grad()
            # Predict the noise
            eps_hat = model(z)
            # Compute the loss
            loss = criterion(eps_hat, eps)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    total_val_samples = 0
    with t.no_grad():
        for inputs in val_loader:
            batch_x = inputs[0]
            # Add noise to the validation data
            z, eps = noisy_channels(batch_x, t.ones((len(batch_x))) * logsnr)
            eps_hat = model(z)
            val_loss += criterion(eps_hat, eps).item()
            total_val_samples += len(batch_x)

    val_loss /= total_val_samples

    return val_loss

def svd_predict_and_evaluate(x, logsnr, batch_size, rank):
    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0
    dataset = TensorDataset(x)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    for batch in data_loader:
        batch_x = batch[0]
        z, eps = noisy_channels(batch_x, t.ones((len(batch_x))) * logsnr)

        for i in range(z.size(0)):
            u, s, vt = t.linalg.svd(z[i], full_matrices=False)

            num_singular_values_to_keep = rank

            s_hat = s[:num_singular_values_to_keep]
            u_hat = u[:, :num_singular_values_to_keep]
            vt_hat = vt[:num_singular_values_to_keep, :]

            z_hat = t.matmul(u_hat, t.matmul(t.diag(s_hat), vt_hat))

            eps_hat = z[i] - z_hat

            loss = criterion(eps_hat, eps[i])
            total_loss += loss.item()

    avg_loss = total_loss / len(dataset)
    return avg_loss

# Define training parameters
batch_size = 50
num_epochs_linear = 1
num_epochs_mlp = 1
learning_rate = 0.001
patience = 20
dropout = 0
weight_decay = 0.1
lambda_l1 = 0.1

# Define logsnr range
loc, s = 0, 1
logsnrs = t.linspace(loc-4*s, loc+4*s, 10)

# Initialize lists to store MSE values
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

# rank=1, tail_length=0.1
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
