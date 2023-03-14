import numpy as np

def normalize_data(X, vmin = 0, vmax = 255):
    X_scaled = []
    for i in range(X.shape[0]):
        X_scaled.append((X[i] - vmin)/(vmax - vmin))

    return np.array(X_scaled)


def train_test_split(X, y, val_split_ratio=0.1):
    indices = np.arange(X.shape[0])
    n_train_samples = int(X.shape[0] * (1-val_split_ratio))
    np.random.shuffle(indices)
    return X[indices[: n_train_samples]], y[indices[: n_train_samples]], X[indices[n_train_samples:]], y[indices[n_train_samples:]]
    