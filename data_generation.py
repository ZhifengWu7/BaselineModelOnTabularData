import numpy as np
import torch as t
from sklearn.datasets import make_low_rank_matrix

def generate_tabular_data(n_samples, rows, columns, rank, tail_strength, random_state):
    x = np.empty((n_samples, rows, columns))
    for i in range(n_samples):
        x[i] = make_low_rank_matrix(n_samples=rows, n_features=columns, effective_rank=rank, tail_strength=tail_strength, random_state=random_state+i)

    x_np = np.array(x)
    means = np.mean(x_np, axis=0)
    std_devs = np.std(x_np, axis=0)
    x_normalized = (x_np - means) / std_devs

    return t.tensor(x_normalized, dtype=t.float32)
