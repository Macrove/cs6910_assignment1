import numpy as np

class CrossEntropy():
    def __init__(self):
        pass
    
    def compute(self, Y, probab_matrix):
        return np.sum(Y @ probab_matrix.T)

    # def differentiate(self, Y, )
        