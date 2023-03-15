import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        pass
    
    def compute(self, y, probabs):
        return -np.log(np.sum(y * probabs))

    def grad(self, y, probabs):
        return -(y - probabs)

class SquaredErrorLoss():
    def __init__(self) -> None:
        pass

    def compute(self, y, probabs):
        return np.sum(np.power((y - probabs), 2) / y.shape[0])

    def grad(self, y, probabs):
        return (probabs - y)/ probabs.shape[0]

    
        