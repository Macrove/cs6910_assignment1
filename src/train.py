import argparse

import wandb
from params.default_params import default_credentials, default_dataset, default_model_params, optimizer_param_map
from params.parser_params import parser_choices
from main import main


parser = argparse.ArgumentParser(
                    prog='train',
                    description='Suppy parameters to neural network to run and log results in wandb.ai',
                    epilog="That's all")

optimizer = optimizer_param_map[default_model_params["optimizer"]]

parser.add_argument("-wp", "--wandb_project", choices= parser_choices["wandb_project"], default=default_credentials["wandb_project"])
parser.add_argument("-we", "--wandb_entity", choices=parser_choices["wandb_entity"], default=default_credentials["wandb_entity"])
parser.add_argument("-d", "--dataset", choices=parser_choices["dataset"], default=default_dataset)
parser.add_argument("-e", "--epochs", default=default_model_params["n_epoch"], type=int)
parser.add_argument("-b", "--batch_size", default=default_model_params["batch_size"], type=int)
parser.add_argument("-l", "--loss", choices= parser_choices["loss"], default=default_model_params["loss_func"])
parser.add_argument("-o", "--optimizer", choices=parser_choices["optimizer"], default=default_model_params["optimizer"])
parser.add_argument("-lr", "--learning_rate", default=optimizer["default_params"]["eta"], type=float)
parser.add_argument("-m", "--momentum", default=optimizer_param_map["momentum"]["default_params"]["gamma"], type=float)
parser.add_argument("-beta", "--beta", default=optimizer_param_map["rmsprop"]["default_params"]["beta"], type=float)
parser.add_argument("-beta1", "--beta1", default=optimizer_param_map["nadam"]["default_params"]["beta1"], type=float)
parser.add_argument("-beta2", "--beta2", default=optimizer_param_map["nadam"]["default_params"]["beta2"], type=float)
parser.add_argument("-eps", "--epsilon", default=optimizer_param_map["nadam"]["default_params"]["epsilon"], type=float)
parser.add_argument("-w_d", "--weight_decay", default=default_model_params["weight_decay"], type=float)
parser.add_argument("-w_i", "--weight_init", choices=parser_choices["initialization"], default=default_model_params["weight_initialization"])
parser.add_argument("-nhl", "--num_layers", default=default_model_params["n_hidden_layers"], type=int)
parser.add_argument("-sz", "--hidden_size", default=default_model_params["size_hidden_layer"], type=int)
parser.add_argument("-a", "--activation", choices=parser_choices["activation"], default=default_model_params["activation_func"])
parser.add_argument("-wb", "--use_wandb", choices=[0, 1], default=1, type=int)

args = parser.parse_args()

args.eta = args.learning_rate
args.gamma = args.momentum

optimizer = optimizer_param_map[args.optimizer]
for key in optimizer["default_params"].keys():
    optimizer["default_params"][str(key)] = getattr(args, str(key))

print(args)
dataset = args.dataset
n_epoch = args.epochs
n_hidden_layers = args.num_layers
size_hidden_layer = args.hidden_size
weight_decay = args.weight_decay
batch_size = args.batch_size
weight_initialization = args.weight_init
activation_func = args.activation
loss_func = args.loss
use_wandb = args.use_wandb



if use_wandb:
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    run.name = "hl_{}_bs_{}_ac_{}_opt_{}".format(args.num_layers, args.batch_size, args.activation, args.optimizer)
    run.log_code()

main(loss_func, dataset, optimizer, n_epoch, n_hidden_layers, size_hidden_layer, weight_decay, batch_size, weight_initialization, activation_func, use_wandb)