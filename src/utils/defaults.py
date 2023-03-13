
optimizers_map = {
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
            "eta" : 0.0001,
            "beta1": 0.9,
            "beta2": 0.6,
            "epsilon": 0.00001
        }
    }
}

default_model_params = {
    "optimizer" : "sgd",
    "n_epoch" : 10,
    "n_hidden_layers" : 3,
    "size_hidden_layer" : 64,
    "weight_decay" : 0.0000005,
    "batch_size" : 1000,
    "weight_initialization" : "Xavier",
    "activation_func" : "sigmoid"
}

default_dataset = 'fashion_mnist'