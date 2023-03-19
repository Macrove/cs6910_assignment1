import numpy as np
import matplotlib.pyplot as plt
import wandb
from utils.prepare_dataset import prepare_dataset

def plot_and_log_images(X, y, label_dict, use_wandb=False):
    if use_wandb:
        wandb.init(project="cs6910-assignment-1", entity="me19b110", name="images log")

    images = []
    labels = []
    for i in range(10):
        images.append(X[np.where(y==i)[0][0]].reshape((28, 28)))
        labels.append(label_dict[i])
    
    fig, ax = plt.subplots(4, 3, figsize=(10,10))
    print("Sample images from training dataset")
    for idx in range(10):
        ax[int(idx/3), idx%3].imshow(images[idx], cmap="gray", vmin="0", vmax="255")
        ax[int(idx/3), idx%3].set_title(labels[idx])
        ax[int(idx/3), idx%3].axis('off')
    
    fig.delaxes(ax[3,1])
    fig.delaxes(ax[3,2])


    if use_wandb:
        wandb.log({"Image Samples": [wandb.Image(img, caption=lbl) for img, lbl in zip(images, labels)]})
        wandb.finish()
    

x_train, y_train, y_train_enc, x_test, y_test, y_test_enc, label_dict = prepare_dataset()

plot_and_log_images(x_train, y_train, label_dict, use_wandb=True)