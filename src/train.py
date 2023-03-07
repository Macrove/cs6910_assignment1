from model import NeuralNetwork
from utils.prepare_dataset import prepare_dataset

x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict = prepare_dataset()

layers = [
    {
        "name": "input_layer",
    },
    {
        "name": "hl1",
        "size": 64,
        "activation_func": "sigmoid"
    },
    {
        "name": "hl2",
        "size": 128,
        "activation_func": "sigmoid"
    },
    {
        "name": "output_layer",
        "size": 10,
        "activation_func": "sigmoid"
    }
]
nn = NeuralNetwork(X=x_train, layers=layers)
nn.forward_prop()