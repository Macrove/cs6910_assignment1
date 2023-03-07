import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-1 * x))

def softmax(x):
    terms = np.exp(x)
    sum = np.sum(terms)
    return terms/sum
