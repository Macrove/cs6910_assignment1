import numpy as np
from activation_funcs import sigmoid


class input_layer():
    def __init__(self, size, name = "input_layer"):
        self.h = np.zeros(size)
        self.name = name
        self.size = size

class hidden_layer():
    def __init__(self, name, size, activation_func):
        self.name = name
        self.activation_func = activation_func
        self.size = size
        self.a = np.zeros(self.size)
        self.h = np.zeros(self.size)
    
class output_layer():
    def __init__(self, size, activation_func = "sigmoid", name="output_layer"):
        self.name = name
        self.activation_func = activation_func
        self.size = size
        self.a = np.zeros(self.size)
        self.h = np.zeros(self.size)