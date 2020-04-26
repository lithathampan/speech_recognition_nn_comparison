from glob import glob
import numpy as np
import _pickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style(style='white')
from pprint import pprint

def vizualise_model(filter="",listofmodels=None,title =""):
    # obtain the paths for the saved model history
    all_pickles = sorted(glob("results/*"+filter+"*.pickle"))
    if listofmodels:
        all_pickles = [item for item in all_pickles if item[8:-7] in listofmodels]
    #for item in all_pickles:
        #pprint(pickle.load( open( item, "rb" ) ))
    # extract the name of each model
    model_names = [item[8:-7] for item in all_pickles]
    # extract the loss history for each model
    valid_loss = [pickle.load( open( i, "rb" ) )['val_loss'] for i in all_pickles]
    train_loss = [pickle.load( open( i, "rb" ) )['loss'] for i in all_pickles]
    # save the number of epochs used to train each model
    num_epochs = [len(valid_loss[i]) for i in range(len(valid_loss))]

    fig = plt.figure(figsize=(16,5))

    # plot the training loss vs. epoch for each model
    ax1 = fig.add_subplot(121)
    for i in range(len(all_pickles)):
        ax1.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
                train_loss[i], label=model_names[i])
    # clean up the plot
    ax1.legend()  
    ax1.set_xlim([1, max(num_epochs)])
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(title+'-Training Loss')
    plt.grid(b=True)
    # plot the validation loss vs. epoch for each model
    ax2 = fig.add_subplot(122)
    for i in range(len(all_pickles)):
        ax2.plot(np.linspace(1, num_epochs[i], num_epochs[i]), 
                valid_loss[i], label=model_names[i])
    # clean up the plot
    ax2.legend()  
    ax2.set_xlim([1, max(num_epochs)])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(title+'-Validation Loss')
    plt.grid(b=True)
    plt.show()