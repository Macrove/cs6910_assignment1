# Assignment Repository for CS6910

This repository contains all the code related to assignment 1 of cs6910

## API Reference

### Creating Model
The command below will create the model with the given params.
```python
nn = NeuralNetwork(X_train=x_train_, y_train=y_train_, X_val=x_val_, y_val=y_val_,
                    layers=layers, loss_func=loss_func, batch_size=batch_size,
                    n_epoch=n_epoch, shuffle=True, optimizer=optimizer["name"],
                    optimizer_params=optimizer["default_params"], initialization=weight_initialization, decay=weight_decay, use_wandb=use_wandb)
```

You can find detailed information about format and options for ```layers```, ```loss_func```, ```optimizer```, ```initialization``` below.
Note: Shuffle parameter is to shuffle the training dataset after every epoch for training

### Training Model
The command below will train the model object for the dataset already passed during creation of NeuralNetwork object
```python
nn.fit()
```

### Getting Predictions
The command below will output numpy array containing predicted class for all the test samples passed to the dataset
```python
nn.predict(x_test)
```

### Layers
```layers``` is a list of dict passed as argument to ```NeuralNetwork``` model object
#### Input Layer
```python
layers[0] = {
              "name": "input_layer"
            }
``` 
Only argument input layer needs is the ```name``` of layer

#### Hidden Layer
Note: i is not equal to ```-1``` or ```0```
```python
layers[i] = {
              "name": "hl{}".format(i),
              "size": size_hidden_layer,
              "activation_func": activation_func
            }
``` 
Arguments needed: ```name```, ```size```, ```activation_func```


#### Hidden Layer
Note: i is not equal to ```-1``` or ```0```
```python
layers[-1] = {
              "name": "output_layer",
              "size": y_test_enc[0].shape[0],
              "activation_func": "softmax"
            }
``` 
Arguments needed: ```name```, ```size```, ```activation_func```

## Available options
### Loss functions
```python
    CrossEntropyLoss()
    SquaredErrorLoss()
```

### Activation functions
```python
   Sigmoid()
   Softmax()
   Tanh()
   ReLu()
   LeakyReLu()
   Identity()
```

### Optimizers
```python
    Sgd()
    Momentum()
    Nag()
    Rmsprop()
    Adam()
    Nadam()
```

## User Interface
| Commands | Functions |
| --- | --- |
|```python src/train.py``` | fetches parameters passed by command line and trains the model by calling ```main``` function from ```main.py``` file. This further passes arguments to the model to train it and predict the test accuracy on the test dataset. It can be configured to output Confusion Matrix on this test dataset. For more information about the arguments, refer to ```script_cmd.txt```|
| ```python src/q1_log_samples.py``` | fetches dataset by calling ```prepare_dataset``` function from ```src/utils/prepare_dataset.py``` and plots 1 sample image of every class. It can be configured to log the images to wandb as well by changing flag argument ```use_wandb``` to ```True```. The name prefix ```q1``` denotes the question number in the assignment|
| ```python src/q4_wandb_sweep``` | contains sweep configuration details. You can change parameters in ```sweep_configuration``` dict to run and log results and configuration details to wandb. Edit ```count``` argument in ```wandb.agent(sweep_id=sweep_id, function=run_sweeps, count=2)``` to change the number of sweeps you wish to execute. The code can be edited to perform sweeps on specific optimizers with their specific configurations as well. |
| ```script_cmd.txt``` | contains sample command line arguments that can be passed to ```train.py```. If no arguments are passed, default arguments corresponding to best hyperparameters optimized for ```fashion_mnist``` dataset will be taken |

## Arguments Supported

| Name | Default Value | Description |
| --- | --- | --- |
| -wp, --wandb_project |	cs6910-assignment-1 |	Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity	| me19b110 |	Wandb Entity used to track experiments in the Weights & Biases dashboard |
| -d, --dataset |	fashion_mnist | choices: ["mnist", "fashion_mnist"] |
| -e, --epochs |	10 | Number of epochs to train neural network |
| -b, --batch_size | 32	| Batch size used to train neural network |
| -l, --loss | cross_entropy |	choices: ["mean_squared_error", "cross_entropy"] |
| -o, --optimizer |	nadam |	choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] |
| -lr, --learning_rate |	0.00008 |	Learning rate used to optimize model parameters |
| -m, --momentum | 0.0000039 | Momentum used by momentum and nag optimizers |
| -beta, --beta |	0.75 |	Beta used by rmsprop optimizer |
| -beta1, --beta1	| 0.7803	| Beta1 used by adam and nadam optimizers |
| -beta2, --beta2 |	0.89504 |	Beta2 used by adam and nadam optimizers |
| -eps, --epsilon	| 1e-9 | Epsilon used by optimizers |
| -w_d, --weight_decay | 0 | Weight decay used by optimizers |
| -w_i, --weight_init	| Xavier | choices: ["random", "Xavier"] |
| -nhl, --num_layers | 3 | Number of hidden layers used in feedforward neural network |
| -sz, --hidden_size | 128 | Number of hidden neurons in a feedforward layer |
| -a, --activation | LeakyReLU | choices: ["identity", "sigmoid", "tanh", "ReLU", "LeakyReLU"] |
| -wb, --use_wandb | 1 | choices: [0, 1]  |


## Folders Description
| Important Folders | Description |
| --- | --- |
| ```src``` | contains all the source code related to the assignment project |
| ```src/nn``` | contains all the implementation detail of the neural network |
| ```src/utils``` | contains all the utility functions like accuracy metrics, confusion matrix and functions necessary for data preprocessing |
| ```src/params``` | contains default parameters (set to best hyperparameter configurations) and default choices of parser |



    

## Branch Workflow 
  ### Description of branches
  - ```main```: Final submission will be done via this branch.  
  - ```dev```: Once the features/questions are completed, they will be merged to this branch. 
  - ```q<n>```: individuals questions will be completed in these branches. n represents question number. 


## Contributors

student name: HARSHIT RAJ  
email: me19b110@smail.iitm.ac.in  
 
course: CS6910 - FUNDAMENTALS OF DEEP LEARNING  
professor: DR. MITESH M. KHAPRA  
 
ta: ASHWANTH KUMAR  
email: cs21m010@smail.iitm.ac.in   
