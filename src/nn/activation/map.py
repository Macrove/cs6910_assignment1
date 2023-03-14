from nn.activation.functions import Sigmoid, Softmax, ReLu, Identity, Tanh

activation_func_map = {
    "sigmoid": Sigmoid(),
    "softmax": Softmax(),
    "ReLU": ReLu(),
    "identity": Identity(),
    "tanh": Tanh()
}