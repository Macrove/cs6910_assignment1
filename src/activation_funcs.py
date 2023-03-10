import numpy as np

class sigmoid():
    def __init__(self):
        pass
    
    def compute(self, x):
        return 1/(1+np.exp(-1 * x))

    def grad(self, x):
        comp_x = self.compute(x)
        return comp_x * (1 - comp_x)

class softmax():
    def __init__(self):
        pass
    
    def compute(self, x):

        # terms = np.exp(x - np.max(x))
        # print(np.max(terms))
        # print("x", x, "terms", terms)
        terms = np.exp(x)
        return terms/np.sum(terms)
    
    def grad(self, x):
        pass
    
class tanh():
    def __init__(self) -> None:
        pass
