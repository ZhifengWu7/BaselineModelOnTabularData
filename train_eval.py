import torch as t
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn

def noisy_channels(x, logsnr):
    dims = tuple(1 for _ in range(len(x[0].shape)))
    left = (-1,) + dims
    logsnr = logsnr.view(left)
    eps = t.randn((len(logsnr),)+x[0].shape)
    return t.sqrt(t.sigmoid(logsnr))*x + t.sqrt(t.sigmoid(-logsnr))*eps, eps

def train_and_evaluate(x, logsnr, model, batch_size, num_epochs, learning_rate, weight_decay):
    dataset = TensorDataset(x)
    n = len(dataset)
    train_size = int(0.9 * n)
    val_size = n - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='sum')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch_x = batch[0]
            z, eps = noisy_channels(batch_x, t.ones(len(batch_x)) * logsnr)
            optimizer.zero_grad()
            eps_hat = model(z)
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
            z, eps = noisy_channels(batch_x, t.ones((len(batch_x))) * logsnr)
            eps_hat = model(z)
            val_loss += criterion(eps_hat, eps).item()
            total_val_samples += len(batch_x)
    val_loss /= total_val_samples

    return val_loss

def compute_baseline_mse(x, logsnrs):
    criterion = nn.MSELoss(reduction='sum')
    baseline_mses = []
    for logsnr in logsnrs:
        logsnr = t.ones(len(x)) * logsnr
        z, eps = noisy_channels(x, logsnr)
        dims = tuple(1 for _ in range(len(x[0].shape)))
        left = (-1,) + dims
        logsnr = logsnr.view(left)
        eps_hat = t.sqrt(t.sigmoid(-logsnr)) * z
        baseline_mse = criterion(eps_hat, eps).item() / len(x)
        baseline_mses.append(baseline_mse)
    return baseline_mses
