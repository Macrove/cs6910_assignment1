import numpy as np

class CrossEntropy():
    def __init__(self):
        pass
    
    def compute(self, y, probabs):
        return -np.sum(y * probabs)

    # def differentiate(self, Y, )
        