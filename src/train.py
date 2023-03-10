from model import NeuralNetwork
from utils.prepare_dataset import prepare_dataset
from utils.train_test_split import train_test_split
import numpy as np
from utils.normalize_data import normalize_data

x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict = prepare_dataset()
x_train = normalize_data(x_train, vmin=0, vmax=255)
x_test = normalize_data(x_test, vmin=0, vmax=255)

    
x_train_, y_train_, x_val_, y_val_ = train_test_split(x_train, y_train_enc, 0.1)
layers = [
    {
        "name": "input_layer",
    },
    {
        "name": "hl1",
        "size": 128,
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
        "activation_func": "softmax"
    }
]
optimizers = {
    "batch_gradient_descent" : {
        "name": "batch_gradient_descent",
        "params": {
            "eta" : 0.0001
        }
    }
}
optimizer = optimizers["batch_gradient_descent"]
nn = NeuralNetwork(X_train=x_train_, y_train=y_train_, X_val=x_val_, y_val=y_val_,
                   layers=layers, loss_func="cross_entropy", batch_size=1000,
                   n_epoch=10, shuffle=True, optimizer=optimizer["name"], optimizer_params=optimizer["params"])
nn.fit()