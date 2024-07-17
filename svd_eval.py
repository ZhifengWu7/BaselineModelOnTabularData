import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def noisy_channels(x, logsnr):
    dims = tuple(1 for _ in range(len(x[0].shape)))
    left = (-1,) + dims
    logsnr = logsnr.view(left)
    eps = t.randn((len(logsnr),)+x[0].shape)
    return t.sqrt(t.sigmoid(logsnr))*x + t.sqrt(t.sigmoid(-logsnr))*eps, eps

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
