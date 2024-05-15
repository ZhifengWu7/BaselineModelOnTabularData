import torch as t
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_low_rank_matrix
from torch.utils.data import TensorDataset, DataLoader, random_split

def make_low_rank_matrix(n, m, rank):
    """Generate a low-rank matrix of shape (n, m) and given rank."""
    U = np.random.randn(n, rank)
    V = np.random.randn(rank, m)
    return np.dot(U, V)
def sample_low_rank_matrices(num_samples, n, m, rank):
    """Generate a specified number of low-rank matrices."""
    matrices = [make_low_rank_matrix(n, m, rank) for _ in range(num_samples)]
    return matrices

# Generate tabular data
n_samples = 100  # Number of samples
n = 50  # Number of rows in each matrix
m = 30  # Number of columns in each matrix
rank = 20  # Rank of the matrices

# Generate tabular data
x_origin = sample_low_rank_matrices(num_samples=n_samples, n=n, m=m, rank=rank)
X_array = np.array(x_origin)
X = t.tensor(X_array, dtype=t.float32)

def noisy_channels(x, logsnr):
    """Add Gaussian noise to x, return "z" and epsilon."""
    dims = tuple(1 for _ in range(len(x[0].shape)))
    left = (-1,) + dims
    logsnr = logsnr.view(left)
    eps = t.randn((len(logsnr),)+x[0].shape)
    return t.sigmoid(logsnr)*x + t.sigmoid(-logsnr)*eps, eps

class LinearRegressionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegressionModel, self).__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(x.size(0), *self.out_dim)

# Define a linear regression model class.
# - in_dim: Input dimension of the data.
# - out_dim: Output dimension of the data.
# The model consists of a single linear layer that transforms the input data.
# The forward method reshapes the input, applies the linear layer, and reshapes the output to the desired dimensions.

class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=50, n_layers=2, out_dim=(50,30), activation=nn.ReLU, dropout=0.):
        super(MLPModel, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(activation())
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        for _ in range(n_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim,in_dim))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x.view(x.size(0),*self.out_dim)

# Define a multi-layer perceptron (MLP) model class.
# - in_dim: Input dimension of the data.
# - hidden_dim: Dimension of the hidden layers (default: 50).
# - n_layers: Number of layers in the model (default: 2).
# - out_dim: Output dimension of the data (default: (50, 30)).
# - activation: Activation function to use (default: ReLU).
# - dropout: Dropout rate (default: 0).
# The model consists of an input layer, several hidden layers with activation and optional dropout, and an output layer.
# The forward method reshapes the input, applies each layer in sequence, and reshapes the output to the desired dimensions.

def train_and_evaluate(x, logsnr, model, batch_size, num_epochs):
    """Train a model with the given dataset and return the validation loss."""
    z, noise = noisy_channels(x, t.ones(len(x))*logsnr)
    dataset = TensorDataset(z, noise)
    n = len(dataset)

    # Split the dataset into training and validation sets
    train_size = int(0.9 * n)
    val_size = n - train_size
    train_dataset, val_dataset = t.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Setup the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate the model after training is complete
    model.eval()
    val_loss = 0
    with t.no_grad():
      for inputs, targets in val_loader:
        outputs = model(inputs)
        val_loss += criterion(outputs, targets).item() / len(val_loader)
    return val_loss

# Train the two model
batch_size = 5
num_epochs = 500

loc, s = 0.0, 1.0
logsnrs = t.linspace(loc - 4 * s, loc + 4 * s, 20)

mmse_g = m*n*t.sigmoid(logsnrs)

linear_mses = []
linear_model = LinearRegressionModel(in_dim=n*m, out_dim=(n,m))

mlp_mses = []
mlp_model = MLPModel(in_dim=n*m, hidden_dim=50, n_layers=4, out_dim=(n,m), dropout=0.1)

for logsnr in logsnrs:
  # val_loss_linear = train_and_evaluate(x=X, logsnr=logsnr, model=linear_model, batch_size=batch_size,num_epochs=num_epochs)
   val_loss_mlp = train_and_evaluate(x=X, logsnr=logsnr, model=mlp_model, batch_size=batch_size,num_epochs=num_epochs)
  # linear_mses.append(val_loss_linear)
   mlp_mses.append(val_loss_mlp)

linear_mses = [x * m * n for x in linear_mses]

mlp_mses = [x * m * n for x in mlp_mses]

fig, ax = plt.subplots(1, 1)
ax.plot(logsnrs, linear_mses, label="Linear_Model")
ax.plot(logsnrs, mlp_mses, label="MLP_Model")
ax.plot(logsnrs, mmse_g, label="MMSE Gaussian")
ax.set_ylabel('$E[(\epsilon - \hat \epsilon)^2]$')
ax.set_xlabel('log SNR ($\\alpha$)')
ax.legend()
plt.title('Comparing MLP, Linear Model, and MMSE Gaussian')
plt.show()
