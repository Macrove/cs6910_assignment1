from model import NeuralNetwork
from utils.prepare_dataset import prepare_dataset
from utils.train_test_split import train_test_split
import numpy as np
from utils.normalize_data import normalize_data
from utils.defaults import optimizers_map, default_model_params, default_dataset
import wandb


def main(use_wandb = True):


    optimizer = optimizers_map[default_model_params["optimizer"]]
    n_epoch = default_model_params["n_epoch"]
    n_hidden_layers = default_model_params["n_hidden_layers"]
    size_hidden_layer = default_model_params["size_hidden_layer"]
    weight_decay = default_model_params["weight_decay"]
    batch_size = default_model_params["batch_size"]
    weight_initialization = default_model_params["weight_initialization"]
    activation_func = default_model_params["activation_func"]
    
    if(use_wandb):
        run = wandb.init()
        config = wandb.config
        run.name = "hl_{}_bs_{}_ac_{}_opt_{}".format(config.n_hidden_layers, config.batch_size, config.activation_func, config.optimizer)

        optimizer = optimizers_map[config.optimizer]
        for key in optimizer["default_params"].keys():
            optimizer["default_params"][str(key)] = getattr(config, str(key))

        n_epoch = config.n_epoch
        n_hidden_layers = config.n_hidden_layers
        size_hidden_layer = config.size_hidden_layer
        weight_decay = config.weight_decay
        batch_size = config.batch_size
        weight_initialization = config.weight_initialization
        activation_func = config.activation_func

    dataset_type = default_dataset

    x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict = prepare_dataset(dataset_type, normalize = True)
    
    #splitting dataset
    x_train_, y_train_, x_val_, y_val_ = train_test_split(x_train, y_train_enc, 0.1)
    
    #creating layers
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
        "size": y_test_enc[0].shape[0],
        "activation_func": "softmax"
        
    })

    nn = NeuralNetwork(X_train=x_train_, y_train=y_train_, X_val=x_val_, y_val=y_val_,
                    layers=layers, loss_func="cross_entropy", batch_size=batch_size,
                    n_epoch=n_epoch, shuffle=True, optimizer=optimizer["name"],
                    optimizer_params=optimizer["default_params"], initialization=weight_initialization, decay=weight_decay, use_wandb=use_wandb)
    nn.fit()


# main(use_wandb=False)