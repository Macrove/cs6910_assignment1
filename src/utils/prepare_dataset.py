import numpy as np
from keras.datasets import fashion_mnist, mnist
from utils.preprocess import normalize_data

# function to prepare dataset
def prepare_dataset(dataset_type: str, normalize: bool = False):
    
    print("Downloading {} dataset:".format(dataset_type.upper()), end=" ")
    if dataset_type == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset_type == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("done")

    print("Samples in training data:\t", x_train.shape[0])
    print("Samples in testing data:\t", x_test.shape[0])
    label_dict = {
        0: 	"T-shirt/top",
        1: 	"Trouser",
        2: 	"Pullover",
        3: 	"Dress",
        4: 	"Coat",
        5: 	"Sandal",
        6: 	"Shirt",
        7: 	"Sneaker",
        8: 	"Bag",
        9: 	"Ankle boot "
    }
    print("Encoding of labels", label_dict)
    
    print("Reshaping images:", end=" ")
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print("done")

    print("Performing one hot encoding:", end=" ")
    y_train_enc = np.zeros((y_train.shape[0], 10))
    y_test_enc = np.zeros((y_test.shape[0], 10))
    print("done")

    for idx, lbl in enumerate(y_train):
        y_train_enc[idx][lbl] = 1
        
    for idx, lbl in enumerate(y_test):
        y_test_enc[idx][lbl] = 1 

    if normalize_data:
        #normalization
        x_train = normalize_data(x_train, vmin=0, vmax=255)
        x_test = normalize_data(x_test, vmin=0, vmax=255)

    print("Dataset Prepared")
    return x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict

