import numpy as np
from layer import input_layer, hidden_layer, output_layer
from activation_funcs import softmax
from loss_funcs import CrossEntropy
from copy import deepcopy
from optimizers import batch_gradient_descent
from utils.accuracy import accuracy
from activation_funcs import sigmoid, softmax

loss_funcs = {
        "cross_entropy": CrossEntropy()
    }

optimizers = {
        "batch_gradient_descent": batch_gradient_descent()
    }

activation_funcs = {
    "sigmoid": sigmoid(),
    "softmax": softmax()
}

class NeuralNetwork():
    def __init__(self, X_train, y_train, X_val, y_val, layers, loss_func, batch_size, n_epoch, shuffle, optimizer, optimizer_params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.y_train_non_enc = np.array([np.argmax(y_train[idx]) for idx in range(y_train.shape[0])])
        self.y_val_non_enc = np.array([np.argmax(y_val[idx]) for idx in range(y_val.shape[0])])
        self.initialize_layers(X_train.shape[1], layers)
        self.loss_func = deepcopy(loss_funcs[loss_func])
        self.batch_size = batch_size
        self.n_epochs = n_epoch
        self.shuffle = shuffle
        self.optimizer = deepcopy(optimizers[optimizer])
        self.optimizer.set_params(optimizer_params)
        print("Training data shape - X_train - {}, y_train - {}".format(X_train.shape, y_train.shape))
        print("Validation data shape - X_val - {}, y_val - {}".format(X_val.shape, y_val.shape))

    def initialize_layers(self, input_size, layers):
        np.random.seed(42)
        
        # self.layers = [input_layer, hidden_layer_1, hidden_layer_2, ..., output_layer]
        print("Initializing layers")
        self.layers = []
        print("layers:")
        
        #input layer
        self.layers.append(input_layer(input_size, layers[0]['name']))
        
        #hidden layers
        layer_idx = 1
        for layer in layers[1:-1]:
            print(layer)
            if layer['name']:
                name = layer['name']
            else:
                name = "hidden_layer_{}".format(layer_idx)
            
            self.layers.append(hidden_layer(name, layer['size'], deepcopy(activation_funcs[layer['activation_func']])))
            self.layers[layer_idx].w = np.random.normal(loc=0,scale=1.0, size=(self.layers[layer_idx].size, self.layers[layer_idx - 1].size))/100
            self.layers[layer_idx].b = np.random.normal(loc=0, scale=1.0, size=self.layers[layer_idx].size)/100

            layer_idx += 1
        
        #output layer
        self.layers.append(output_layer(layers[-1]['size'], deepcopy(activation_funcs[layers[-1]['activation_func']]), layers[-1]['name']))
        self.layers[-1].w = np.random.normal(loc=0, scale=1.0, size=(self.layers[-1].size, self.layers[-2].size))/100
        self.layers[-1].b = np.random.normal(loc=0, scale=1.0, size=self.layers[-1].size)/100

        print("Neural Network prepared\n")


    def forward_prop(self, x):
        
        #initializing input layer
        self.layers[0].h = x

        # propagation
        for idx in range(1,len(self.layers)):
            self.layers[idx].a = self.layers[idx].w @ self.layers[idx - 1].h + self.layers[idx].b
            self.layers[idx].h = self.layers[idx].activation_func.compute(self.layers[idx].a)

        return self.layers[-1].h


    def backward_prop(self, y):

        #compute output gradient
        self.layers[-1].a_grad = -(y - self.layers[-1].h)
        for k in range(len(self.layers)-1, 0, -1):

            #compute gradient wrt parameters    
            self.layers[k].w_grad = np.outer(self.layers[k].a_grad, self.layers[k-1].h)
            self.layers[k].b_grad = self.layers[k].a_grad

            if(k > 1):
                #compute gradient wrt layer below
                self.layers[k-1].h_grad = self.layers[k].w.T @ self.layers[k].a_grad

                #compute gradient wrt layer below (pre - activation)
                self.layers[k-1].a_grad = self.layers[k-1].h_grad * self.layers[k-1].activation_func.grad(self.layers[k-1].a)

    def fit(self):

        indices = np.arange(0, self.X_train.shape[0], 1)

        for epoch in range(self.n_epochs):
            print("Epoch {} - ".format(epoch), end=" ")
            
            num_batches = int(self.X_train.shape[0]/self.batch_size)

            #shuffled indices of data
            if self.shuffle:
                np.random.shuffle(indices)

            for idx in range(1, len(self.layers)):
                self.layers[idx].del_w = np.zeros(self.layers[idx].w.shape)
                self.layers[idx].del_b = np.zeros(self.layers[idx].b.shape)

            for step in range(num_batches):
                
                #computing gradients
                batch_indices = indices[step * self.batch_size : (step + 1) * self.batch_size]
                for idx in batch_indices:
                    self.forward_prop(self.X_train[idx])
                    self.backward_prop(self.y_train[idx])
                    for idx in range(1, len(self.layers)):
                        self.layers[idx].del_w += self.layers[idx].w_grad
                        self.layers[idx].del_b += self.layers[idx].b_grad
                
                #updating weights
                for idx in range(1, len(self.layers)):
                    w_update, b_update = self.optimizer.get_update(self.layers[idx].del_w, self.layers[idx].del_b)
                    self.layers[idx].w -= w_update
                    self.layers[idx].b -= b_update

            print("===============================================================>\t", end="")

            train_acc, val_acc = self.get_accuracy()
            print("training acc = {}, val_acc = {}".format(round(train_acc, 3), round(val_acc, 3)))


    def get_accuracy(self):
        
        y_train_pred = []
        y_train = self.y_train_non_enc
        for idx in range(self.X_train.shape[0]):
            probabs = self.forward_prop(self.X_train[idx])     
            y_train_pred.append(np.argmax(probabs))

        y_train_pred = np.array(y_train_pred)
        train_acc = accuracy(y_train_pred, y_train)
        
        y_val_pred = []
        y_val = self.y_val_non_enc
        for idx in range(self.X_val.shape[0]):
            probabs = self.forward_prop(self.X_val[idx])     
            y_val_pred.append(np.argmax(probabs))

        y_val_pred = np.array(y_val_pred)
        val_acc = accuracy(y_val_pred, y_val)

        return train_acc, val_acc