from model import NeuralNetwork
from utils.prepare_dataset import prepare_dataset
from utils.train_test_split import train_test_split
import numpy as np
from utils.normalize_data import normalize_data
import wandb


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


def main(use_wandb = True):

    
    optimizer = optimizers["sgd"]
    optimizer["params"]["eta"] = 0.0001
    n_epoch = 10
    n_hidden_layers = 3
    size_hidden_layer = 64
    weight_decay = 0.0000005
    batch_size = 1000
    weight_initialization = "Xavier"
    activation_func = "sigmoid"
    
    if(use_wandb):
        run = wandb.init(project="cs6910-assignment-1", entity="me19b110")

        optimizer = optimizers[wandb.config.optimizer]
        optimizer["params"]["eta"] = wandb.config.eta
        n_epoch = wandb.config.n_epoch
        n_hidden_layers = wandb.config.n_hidden_layers
        size_hidden_layer = wandb.config.size_hidden_layer
        weight_decay = wandb.config.weight_decay
        batch_size = wandb.config.batch_size
        weight_initialization = wandb.config.weight_initialization
        activation_func = wandb.config.activation_func

    else:
        pass
    
    dataset_type = 'fashion_mnist'

    x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict = prepare_dataset(dataset_type)
    
    #normalization
    x_train = normalize_data(x_train, vmin=0, vmax=255)
    x_test = normalize_data(x_test, vmin=0, vmax=255)
    
    #splitting dataset
    x_train_, y_train_, x_val_, y_val_ = train_test_split(x_train, y_train_enc, 0.1)
    
    layers = [
        {
            "name": "input_layer",
        }
    ]
    
    for idx in range(n_hidden_layers):
        layers.append({
            "name": "hl{}".format(idx),
            "size": size_hidden_layer,
            "activation_func": activation_func
        })

    #output layer
    layers.append({
        "name": "output_layer",
        "size": 10,
        "activation_func": "softmax"
        
    })

    nn = NeuralNetwork(X_train=x_train_, y_train=y_train_, X_val=x_val_, y_val=y_val_,
                    layers=layers, loss_func="cross_entropy", batch_size=1000,
                    n_epoch=n_epoch, shuffle=True, optimizer=optimizer["name"],
                    optimizer_params=optimizer["params"], initialization=weight_initialization, decay=weight_decay, use_wandb=True)
    nn.fit()


# main()