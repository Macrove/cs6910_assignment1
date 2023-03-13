import wandb
from train import main

sweep_configuration = {
    "name": "question_4_sweep",
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'val_acc'},
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 5
     },
    "parameters": {
        'n_epoch': {'values': [8, 9, 10, 11, 12]},
        'n_hidden_layers': {'values': [2, 3, 4, 5]},
        'size_hidden_layer': {'values': [32, 64]},
        'weight_decay': {'values': [0, 1e-6, 1e-7, 1e-8]},
        'eta': {'min': 1 * 1e-5, 'max': 1 * 1e-3},
        'optimizer': {'values' :['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64, 128]},
        'weight_initialization': {'values': ['random', 'Xavier']},
        'activation_func': {'values': ['sigmoid', 'ReLU']},
        'beta': {'min': 0.5, 'max': 0.99},
        'beta1': {'min': 0.5, 'max': 0.99},
        'beta2': {'min': 0.5, 'max': 0.99},
        'epsilon': {'min': 1e-8, 'max': 1e-4}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-1", entity="me19b110")

wandb.agent(sweep_id=sweep_id, function=main, count=200)
