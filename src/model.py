import numpy as np
from layer import input_layer, hidden_layer, output_layer
from activation_funcs import softmax

class NeuralNetwork():
    def __init__(self, X, layers):
        self.X = X
        self.__initialize_layers(X.shape[1], layers)

    def __initialize_layers(self, input_size, layers):
        
        # self.layers = [input_layer, hidden_layer_1, hidden_layer_2, ..., output_layer]
        print("Initializing layers")
        self.__layers = []
        # print(layers)
        #input layer
        self.__layers.append(input_layer(input_size, layers[0]['name']))
        
        #hidden layers
        layer_idx = 1
        for layer in layers[1:-1]:
            print(layer)
            if layer['name']:
                name = layer['name']
            else:
                name = "hidden_layer_{}".format(layer_idx)
            
            self.__layers.append(hidden_layer(name, layer['size'], layer['activation_func']))
            self.__layers[layer_idx].W = np.random.rand(self.__layers[layer_idx].size, self.__layers[layer_idx - 1].size)
            self.__layers[layer_idx].b = np.random.rand(self.__layers[layer_idx].size)

            layer_idx += 1
        
        #output layer
        self.__layers.append(output_layer(layers[-1]['size'], layers[-1]['activation_func'], layers[-1]['name']))
        self.__layers[-1].W = np.random.rand(self.__layers[-1].size, self.__layers[-2].size)
        self.__layers[-1].b = np.random.rand(self.__layers[-1].size)

        print("Neural Network prepared\n")

    def forward_prop(self):
        
        y = np.zeros((self.X.shape[0], self.__layers[-1].size))
        for idx, x in enumerate(self.X):
            self.__layers[0].a = x
            layer_idx = 1
            for layer in self.__layers[1:]:
                # print(layer.W.shape, self.__layers[layer_idx - 1].a.shape, layer.b.shape,layer_idx)
                layer.a = layer.W @ self.__layers[layer_idx - 1].a + layer.b
                layer.h = layer.activate(layer.a)
                layer_idx+=1

            y[idx] = softmax(layer.h)
        
        print(y.shape)
        print(y)


        
    