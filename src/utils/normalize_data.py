import numpy as np

def normalize_data(X, vmin = 0, vmax = 255):
    X_scaled = []
    for i in range(X.shape[0]):
        X_scaled.append((X[i] - vmin)/(vmax - vmin))

    return np.array(X_scaled)