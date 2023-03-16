import numpy as np

class Sigmoid():
    def __init__(self):
        pass
    
    def compute(self, x):
        return 1/(1+np.exp(-1 * x))

    def grad(self, x):
        comp_x = self.compute(x)
        return comp_x * (1 - comp_x)

class Softmax():
    def __init__(self):
        pass
    
    def compute(self, x):
        x -= np.max(x)
        terms = np.exp(x)
        return terms/np.sum(terms)
    
    def grad(self, x):
        pass
    
class Tanh():
    def __init__(self):
        pass

    def compute(self, x):
        terms = np.exp(x)
        return (terms - 1/terms)/(terms + 1/terms)

    def grad(self, x):
        comp_x = self.compute(x)
        return 1 - np.power(comp_x, 2)


class ReLu():
    def __init__(self):
        pass
    
    def compute(self, x):
        return np.array([t if t>0 else 0 for t in x])

    def grad(self, x):
        return np.array([0 if t < 0 else 1 for t in x])

class Identity():
    def __init__(self) -> None:
        pass

    def compute(self, x):
        return x

    def grad(self, x):
        return 1
