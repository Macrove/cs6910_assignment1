import math
import numpy as np
from nn.layer.layer import Input_layer, Hidden_layer, Output_layer
from copy import deepcopy
from utils.metrics import accuracy
import wandb
from nn.loss.map import loss_func_map
from nn.optimizer.map import optimizer_map
from nn.activation.map import activation_func_map
import time

class NeuralNetwork():
    def __init__(self, X_train, y_train, X_val, y_val, layers, loss_func, batch_size, n_epoch, shuffle, optimizer, optimizer_params, initialization, decay, use_wandb):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.y_train_non_enc = np.array([np.argmax(y_train[idx]) for idx in range(y_train.shape[0])])
        self.y_val_non_enc = np.array([np.argmax(y_val[idx]) for idx in range(y_val.shape[0])])
        self.initialization = initialization
        self.loss_func = deepcopy(loss_func_map[loss_func])
        self.batch_size = batch_size
        self.n_epochs = n_epoch
        self.shuffle = shuffle
        self.decay = decay
        self.use_wandb = use_wandb
        self.initialize_layers(X_train.shape[1], layers, optimizer, optimizer_params)
        print("Neural Network prepared\n")
        print("Training data shape - X_train - {}, y_train - {}".format(X_train.shape, y_train.shape))
        print("Validation data shape - X_val - {}, y_val - {}".format(X_val.shape, y_val.shape))

    def initialize_layers(self, input_size, layers, optimizer, optimizer_params):
        np.random.seed(42)
        
        # self.layers = [input_layer, hidden_layer_1, hidden_layer_2, ..., output_layer]
        self.layers = []
        print("hidden layers:")
        
        #input layer
        self.layers.append(Input_layer(input_size, layers[0]['name']))
        
        #hidden layers
        layer_idx = 1
        for layer in layers[1:-1]:
            print(layer)
            if layer['name']:
                name = layer['name']
            else:
                name = "hidden_layer_{}".format(layer_idx)
            
            self.layers.append(Hidden_layer(name, layer['size'], deepcopy(activation_func_map[layer['activation_func']])))
            if self.initialization == "random":
                self.layers[layer_idx].w = np.random.normal(loc=0,scale=1.0, size=(self.layers[layer_idx].size, self.layers[layer_idx - 1].size))/100
                self.layers[layer_idx].b = np.random.normal(loc=0, scale=1.0, size=self.layers[layer_idx].size)/100
            elif self.initialization == "Xavier":
                self.layers[layer_idx].w = np.random.normal(loc=0,scale=np.power(2/(self.layers[layer_idx].size + self.layers[layer_idx-1].size),0.5), size=(self.layers[layer_idx].size, self.layers[layer_idx - 1].size))
                self.layers[layer_idx].b = np.random.normal(loc=0,scale=np.power(2/(self.layers[layer_idx].size + self.layers[layer_idx-1].size),0.5), size=self.layers[layer_idx].size)
                
            self.layers[layer_idx].optimizer = deepcopy(optimizer_map[optimizer])
            self.layers[layer_idx].optimizer.set_params({**optimizer_params,
                                                         "w_shape": self.layers[layer_idx].w.shape,
                                                         "b_shape": self.layers[layer_idx].b.shape,
                                                         "decay": self.decay})

            layer_idx += 1
        
        #output layer
        self.layers.append(Output_layer(layers[-1]['size'], deepcopy(activation_func_map[layers[-1]['activation_func']]), layers[-1]['name']))
        self.layers[-1].w = np.random.normal(loc=0, scale=1.0, size=(self.layers[-1].size, self.layers[-2].size))/100
        self.layers[-1].b = np.random.normal(loc=0, scale=1.0, size=self.layers[-1].size)/100
        self.layers[-1].optimizer = deepcopy(optimizer_map[optimizer])
        self.layers[-1].optimizer.set_params({**optimizer_params,
                                                        "w_shape": self.layers[-1].w.shape,
                                                        "b_shape": self.layers[-1].b.shape})



    def forward_prop(self, x):
        
        #initializing input layer
        self.layers[0].h = x

        # propagation
        for idx in range(1,len(self.layers)):
            self.layers[idx].a = self.layers[idx].w @ self.layers[idx - 1].h + self.layers[idx].b
            self.layers[idx].h = self.layers[idx].activation_func.compute(self.layers[idx].a)

        return self.layers[-1].h


    def backward_prop(self, y):

        self.layers[-1].a_grad = self.loss_func.grad_wrt_a(y, self.layers[-1].h)

        for k in range(len(self.layers)-1, 0, -1):

            #compute gradient wrt parameters    
            self.layers[k].w_grad = np.outer(self.layers[k].a_grad, self.layers[k-1].h) + self.decay * self.layers[k].w
            self.layers[k].b_grad = self.layers[k].a_grad + self.decay * self.layers[k].b

            if(k > 1):
                #compute gradient wrt layer below
                self.layers[k-1].h_grad = self.layers[k].w.T @ self.layers[k].a_grad

                #compute gradient wrt layer below (pre - activation)
                self.layers[k-1].a_grad = self.layers[k-1].h_grad * self.layers[k-1].activation_func.grad(self.layers[k-1].a)

    def fit(self):

        indices = np.arange(0, self.X_train.shape[0], 1)
        for epoch in range(self.n_epochs):
            start = time.time()
            
            num_batches = int(self.X_train.shape[0]/self.batch_size)

            #shuffled indices of data
            if self.shuffle:
                np.random.shuffle(indices)

            for idx in range(1, len(self.layers)):
                self.layers[idx].optimizer.reset()

            for step in range(num_batches):
                
                batch_indices = indices[step * self.batch_size : (step + 1) * self.batch_size]
                
                #partial updates
                for idx in range(1, len(self.layers)):
                    self.layers[idx].w, self.layers[idx].b = self.layers[idx].optimizer.get_partial_update(self.layers[idx].w, self.layers[idx].b)

                for idx in batch_indices[0:]:
                    self.forward_prop(self.X_train[idx])
                    self.backward_prop(self.y_train[idx])
                    for idx in range(1, len(self.layers)):
                        self.layers[idx].optimizer.del_w += self.layers[idx].w_grad
                        self.layers[idx].optimizer.del_b += self.layers[idx].b_grad
                
                #updating weights
                for idx in range(1, len(self.layers)):
                    self.layers[idx].w, self.layers[idx].b = self.layers[idx].optimizer.get_update(self.layers[idx].w, self.layers[idx].b)

            train_acc, val_acc, train_loss, val_loss = self.get_accuracy_and_loss()
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch+1,
                    "training_acc": train_acc,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
            
            if math.isnan(train_loss):
                return

            end = time.time()
            
            print("Epoch {} - ================>\t".format(epoch+1), end=" ")

            print("training acc = {:0.4f}, val_acc = {:0.4f}, train_loss = {:0.4f}, val_loss = {:0.4f}, time_taken = {:0.3f}s".format(train_acc, val_acc, train_loss, val_loss, end - start))

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for idx, x in enumerate(X_test):
            y_pred[idx] = np.argmax(self.forward_prop(x))

        return y_pred


    def get_accuracy_and_loss(self):
        
        y_train_pred = []
        y_train = self.y_train_non_enc
        train_loss = 0
        for idx in range(self.X_train.shape[0]):
            probabs = self.forward_prop(self.X_train[idx])
            train_loss += self.loss_func.compute(self.y_train[idx], probabs)
            y_train_pred.append(np.argmax(probabs))

        y_train_pred = np.array(y_train_pred)
        train_acc = accuracy(y_train_pred, y_train)
        
        
        y_val_pred = []
        y_val = self.y_val_non_enc
        val_loss = 0
        for idx in range(self.X_val.shape[0]):
            probabs = self.forward_prop(self.X_val[idx])     
            val_loss += self.loss_func.compute(self.y_val[idx], probabs)
            y_val_pred.append(np.argmax(probabs))

        y_val_pred = np.array(y_val_pred)
        val_acc = accuracy(y_val_pred, y_val)

        return train_acc, val_acc, train_loss/self.X_train.shape[0], val_loss/self.X_val.shape[0]