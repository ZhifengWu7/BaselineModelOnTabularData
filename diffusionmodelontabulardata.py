import torch as t
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_low_rank_matrix
from torch.utils.data import TensorDataset, DataLoader, random_split

# Generate tabular data
n_samples = 50  # Number of samples
n = 50  # Number of rows in each matrix
m = 30  # Number of columns in each matrix

# Generate tabular data
x = t.randn(n_samples, n, m)

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

def train_and_evaluate(x, logsnr, model, batch_size, num_epochs, learning_rate, patience):
    """Given a specific logsnr value, train the model and return val_loss."""
    dataset = TensorDataset(x)
    n = len(dataset)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * n)
    val_size = n - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Setup the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')

    best_val_loss = float('inf')
    epochs_no_improve = 0

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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    return best_val_loss

# Define training parameters
batch_size = 5
num_epochs_linear = 300
num_epochs_mlp = 50
learning_rate = 0.001
patience = 20

# Define logsnr range
loc, s = 0, 1
logsnrs = t.linspace(loc-4*s, loc+4*s, 10)

# Initialize lists to store MSE values
linear_mses = []
mlp_mses = []
baseline_mses = []
mmse_g = m * n * t.sigmoid(logsnrs)

# Train and evaluate models for each logsnr value
for logsnr in logsnrs:
    linear_model = LinearRegressionModel(in_dim=n*m, out_dim=(n,m))
    mlp_model = MLPModel(in_dim=n*m, out_dim=(n,m), dropout=0)
    val_loss_linear = train_and_evaluate(x=x, logsnr=logsnr, model=linear_model, batch_size=batch_size, num_epochs=num_epochs_linear, learning_rate=learning_rate, patience=patience)
    val_loss_mlp = train_and_evaluate(x=x, logsnr=logsnr, model=mlp_model, batch_size=batch_size, num_epochs=num_epochs_mlp, learning_rate=learning_rate, patience=patience)
    linear_mses.append(val_loss_linear)
    mlp_mses.append(val_loss_mlp)

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

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(logsnrs.numpy(), linear_mses, label='Linear Model')
plt.plot(logsnrs.numpy(), mlp_mses, label='MLP Model')
plt.plot(logsnrs.numpy(), baseline_mses, label='Baseline MSE')
plt.plot(logsnrs.numpy(), mmse_g.numpy(), label='MMSE', linestyle='--')
plt.xlabel('Log SNR')
plt.ylabel('MSE')
plt.title('MSE vs. Log SNR for Multiple Models')
plt.legend()
plt.grid(True)
plt.show()
