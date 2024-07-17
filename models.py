import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearRegressionModel, self).__init__()
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.view(x.size(0), *self.out_dim)

class MLPModel(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c_fc = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        residual = x
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        if self.dropout:
            x = self.dropout(x)
        x += residual
        return x.view(x.size(0), *self.out_dim)
