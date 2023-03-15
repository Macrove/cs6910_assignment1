default_model_params = {
    "optimizer" : "nadam",
    "n_epoch" : 11,
    "n_hidden_layers" : 4,
    "size_hidden_layer" : 64,
    "weight_decay" : 0,
    "batch_size" : 128,
    "weight_initialization" : "Xavier",
    "activation_func" : "ReLU",
    "loss_func": "mean_squared_error"
}

default_dataset = 'fashion_mnist'

default_credentials = {
    "wandb_project": "cs6910-assignment-1",
    "wandb_entity": "me19b110"
}

optimizer_param_map = {
    "sgd" : {
        "name": "sgd",
        "default_params": {
            "eta" : 0.0001
        }
    },
    "momentum" : {
        "name": "momentum",
        "default_params": {
            "eta" : 0.0001,
            "gamma": 0.0001
        }
    },
    "nag" : {
        "name": "nag",
        "default_params": {
            "eta" : 0.0001,
            "gamma": 0.01
        }
    },
    "rmsprop" : {
        "name": "rmsprop",
        "default_params": {
            "eta" : 0.0001,
            "beta": 0.9,
            "epsilon": 0.0001
        }
    },
    "adam" : {
        "name": "adam",
        "default_params": {
            "eta" : 0.0001,
            "beta1": 0.9,
            "beta2": 0.6,
            "epsilon": 0.00001
        }
    },
    "nadam" : {
        "name": "nadam",
        "default_params": {
            "eta" : 0.0002392,
            "beta1": 0.6333,
            "beta2": 0.8638,
            "epsilon": 0.0000312
        }
    }
}