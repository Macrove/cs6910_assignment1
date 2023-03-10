import numpy as np

def accuracy(y_pred, y):
    return np.sum(y_pred == y)/y_pred.shape[0]