import wandb
from main import main
from params.default_params import optimizer_param_map, default_dataset

def run_sweeps():
    run = wandb.init()
    config = wandb.config
    run.name = "hl_{}_bs_{}_ac_{}_opt_{}".format(config.n_hidden_layers, config.batch_size, config.activation_func, config.optimizer)

    optimizer = optimizer_param_map[config.optimizer]
    for key in optimizer["default_params"].keys():
        optimizer["default_params"][str(key)] = getattr(config, str(key))

    n_epoch = config.n_epoch
    n_hidden_layers = config.n_hidden_layers
    size_hidden_layer = config.size_hidden_layer
    weight_decay = config.weight_decay
    batch_size = config.batch_size
    weight_initialization = config.weight_initialization
    activation_func = config.activation_func
    dataset = default_dataset
    loss_func = config.loss

    main(loss_func, dataset, optimizer, n_epoch, n_hidden_layers, size_hidden_layer, weight_decay, batch_size, weight_initialization, activation_func, True)

sweep_configuration = {
    "name": "question_4_efficient_sweep",
    "method": "bayes",
    "metric": {'goal': 'maximize', 'name': 'val_acc'},
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3
     },
    "parameters": {
        'n_epoch': {'values': [8, 9, 10, 11, 12]},
        'n_hidden_layers': {'values': [2, 3, 4, 5]},
        'size_hidden_layer': {'values': [1, 4, 32, 64, 128]},
        'weight_decay': {'values': [0, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
        'eta': {'values': [1e-5, 1e-4, 1e-3, 1e-2, 1e-6]},
        'optimizer': {'values' :['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64, 128]},
        'weight_initialization': {'values': ['random', 'Xavier']},
        'activation_func': {'values': ['sigmoid', 'ReLU', 'LeakyReLU']},
        'beta': {'values': [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]},
        'beta1': {'values': [0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]},
        'beta2': {'values': [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]},
        'epsilon': {'values': [1e-10]},
        'gamma': {'values': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
        'loss': {'values': ['cross_entropy']}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-1", entity="me19b110")
wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=2)

##########################################################################################################
# code for optimizer wise sweeps

# sweep_configuration_map = {
#     "nadam": {
#         "name": "nadam_sweep",
#         "method": "random",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [10, 11, 12, 13, 14]},
#             'n_hidden_layers': {'values': [2, 3, 4]},
#             'size_hidden_layer': {'values': [128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 8 * 1e-5, 'max': 1 * 1e-3},
#             'optimizer': {'values' :['adam']},
#             'batch_size': {'values': [16, 32, 64, 128]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.8},
#             'beta1': {'min': 0.7, 'max': 0.8},
#             'beta2': {'min': 0.8, 'max': 0.95},
#             'epsilon': {'values': [1e-10]},
#             'gamma': {'values': [1e-5]},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "adam": {
#         "name": "adam_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [8, 9, 10, 11, 12, 13, 14]},
#             'n_hidden_layers': {'values': [2, 3, 4, 5]},
#             'size_hidden_layer': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-4},
#             'optimizer': {'values' :['adam']},
#             'batch_size': {'values': [16, 32, 64, 128]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.8},
#             'beta1': {'min': 0.6, 'max': 0.8},
#             'beta2': {'min': 0.7, 'max': 0.9},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "rmsprop": {
#         "name": "rmsprop_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [10, 11, 12, 13]},
#             'n_hidden_layers': {'values': [2, 3, 4, 5]},
#             'size_hidden_layer': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 1e-6, 1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-4},
#             'optimizer': {'values' :['rmsprop']},
#             'batch_size': {'values': [16, 32, 64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.4, 'max': 0.9},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "nag": {
#         "name": "nag_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [10, 11, 12]},
#             'n_hidden_layers': {'values': [2, 3, 4, 5]},
#             'size_hidden_layer': {'values': [64, 128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-5},
#             'optimizer': {'values' :['nag']},
#             'batch_size': {'values': [64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.99},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "momentum": {
#         "name": "momentum_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [8, 9, 10, 11]},
#             'n_hidden_layers': {'values': [2, 3, 4]},
#             'size_hidden_layer': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 1e-6, 1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-4},
#             'optimizer': {'values' :['momentum']},
#             'batch_size': {'values': [16, 32, 64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['LeakyReLU', 'ReLU']},
#             'beta': {'min': 0.5, 'max': 0.99},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'values': [1e-1, 1e-2, 1e-3]},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
#     "sgd": {
#         "name": "sgd_sweep",
#         "method": "bayes",
#         "metric": {'goal': 'maximize', 'name': 'val_acc'},
#         "early_terminate": {
#             "type": "hyperband",
#             "eta": 2,
#             "min_iter": 5
#         },
#         "parameters": {
#             'n_epoch': {'values': [8, 9, 10, 11, 12]},
#             'n_hidden_layers': {'values': [4, 5]},
#             'size_hidden_layer': {'values': [64, 128]},
#             'weight_decay': {'values': [1e-7, 1e-8, 1e-9]},
#             'eta': {'min': 1 * 1e-6, 'max': 1 * 1e-5},
#             'optimizer': {'values' :['sgd']},
#             'batch_size': {'values': [64, 128, 264]},
#             'weight_initialization': {'values': ['Xavier']},
#             'activation_func': {'values': ['ReLU']},
#             'beta': {'min': 0.5, 'max': 0.99},
#             'beta1': {'min': 0.5, 'max': 0.9},
#             'beta2': {'min': 0.7, 'max': 0.99},
#             'epsilon': {'min': 1e-10, 'max': 1e-6},
#             'gamma': {'min': 1e-8, 'max': 1e-5},
#             'loss': {'values': ['cross_entropy']}
#         }
#     },
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration_map["nadam"], project="cs6910-assignment-1", entity="me19b110")
# wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=5)
# for key in sweep_configuration_map.keys():
#     sweep_id = wandb.sweep(sweep=sweep_configuration_map[key], project="cs6910-assignment-1", entity="me19b110")
#     wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=20)

