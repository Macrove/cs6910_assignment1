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
        'epsilon': {'min': 1e-8, 'max': 1e-4},
        'gamma': {'min': 1e-8, 'max': 1e-3},
        'loss': {'values': ['cross_entropy']}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="cs6910-assignment-1", entity="me19b110")

wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=2)

