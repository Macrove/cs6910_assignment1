import wandb
from nn.model import NeuralNetwork
from utils.prepare_dataset import prepare_dataset
from utils.preprocess import train_test_split

def main(loss_func, dataset, optimizer, n_epoch, n_hidden_layers, size_hidden_layer, weight_decay, batch_size, weight_initialization, activation_func, use_wandb):


    x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict = prepare_dataset(dataset, normalize = False, standardize=True)
    
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
                    layers=layers, loss_func=loss_func, batch_size=batch_size,
                    n_epoch=n_epoch, shuffle=True, optimizer=optimizer["name"],
                    optimizer_params=optimizer["default_params"], initialization=weight_initialization, decay=weight_decay, use_wandb=use_wandb)
    nn.fit()

    y_pred = nn.predict(x_test)
    if use_wandb:
        class_names = []
        for key in label_dict.keys():
            class_names.append(label_dict[key])
        wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(
                        y_true=y_test, preds=y_pred,
                        class_names=class_names)})

