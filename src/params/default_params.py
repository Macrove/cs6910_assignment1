default_model_params = {
    "optimizer" : "nadam",
    "n_epoch" : 10,
    "n_hidden_layers" : 3,
    "size_hidden_layer" : 128,
    "weight_decay" : 0,
    "batch_size" : 32,
    "weight_initialization" : "Xavier",
    "activation_func" : "LeakyReLU",
    "loss_func": "cross_entropy"
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
            "eta" : 0.000005618
        }
    },
    "momentum" : {
        "name": "momentum",
        "default_params": {
            "eta" : 0.000001,
            "gamma": 0.0000039
        }
    },
    "nag" : {
        "name": "nag",
        "default_params": {
            "eta" : 0.0000086,
            "gamma": 0.0000021
        }
    },
    "rmsprop" : {
        "name": "rmsprop",
        "default_params": {
            "eta" : 0.000006019,
            "beta": 0.5123,
            "epsilon": 1e-7
        }
    },
    "adam" : {
        "name": "adam",
        "default_params": {
            "eta" : 0.00001,
            "beta1": 0.7483,
            "beta2": 0.7838,
            "epsilon": 1e-9
        }
    },
    "nadam" : {
        "name": "nadam",
        "default_params": {
            "eta" : 0.00008,
            "beta1": 0.7803,
            "beta2": 0.89504,
            "epsilon": 1e-9
        }
    }
}