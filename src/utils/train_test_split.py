import numpy as np

def train_test_split(X, y, val_split_ratio=0.1):
    indices = np.arange(X.shape[0])
    n_train_samples = int(X.shape[0] * (1-val_split_ratio))
    np.random.shuffle(indices)
    return X[indices[: n_train_samples]], y[indices[: n_train_samples]], X[indices[n_train_samples:]], y[indices[n_train_samples:]]
    