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
    "sgd" : {
        "name": "sgd",
        "params": {
            "eta" : 0.0001
        }
    },
    "momentum" : {
        "name": "momentum",
        "params": {
            "eta" : 0.0001,
            "gamma": 0.0001
        }
    },
    "nag" : {
        "name": "nag",
        "params": {
            "eta" : 0.0001,
            "gamma": 0.01
        }
    },
    "rmsprop" : {
        "name": "rmsprop",
        "params": {
            "eta" : 0.0001,
            "beta": 0.9,
            "epsilon": 0.0001
        }
    },
    "adam" : {
        "name": "adam",
        "params": {
            "eta" : 0.0001,
            "beta1": 0.9,
            "beta2": 0.6,
            "epsilon": 0.00001
        }
    },
    "nadam" : {
        "name": "nadam",
        "params": {
            "eta" : 0.0001,
            "beta1": 0.9,
            "beta2": 0.6,
            "epsilon": 0.00001
        }
    }
}
optimizer = optimizers["sgd"]
nn = NeuralNetwork(X_train=x_train_, y_train=y_train_, X_val=x_val_, y_val=y_val_,
                   layers=layers, loss_func="cross_entropy", batch_size=1000,
                   n_epoch=10, shuffle=True, optimizer=optimizer["name"], optimizer_params=optimizer["params"], initialization="Xavier", decay=0.00005)
nn.fit()