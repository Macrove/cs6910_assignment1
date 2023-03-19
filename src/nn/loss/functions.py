import numpy as np

class CrossEntropyLoss():
    def compute(self, y, probabs):
        return -np.log(np.sum(y * probabs) + 1e-100)

    def grad_wrt_a(self, y, probabs):
        return -(y - probabs)

class SquaredErrorLoss():
    def compute(self, y, probabs):
        return np.sum(np.power((y - probabs), 2))/2

    def grad_wrt_a(self, y, probabs):
        terms = probabs * (probabs - y)
        return -probabs * np.sum(terms) + probabs * terms

    
        