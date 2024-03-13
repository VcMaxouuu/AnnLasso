import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest

def draw_curvers(curves_sd, curves_ista):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=500, sharex=True) 
    current_epoch_train = 0
    current_epoch_test = 0

    colors = ['#5A5B9F', '#D94F70', '#009473', '#F0C05A', '#7BC4C4', '#FF6F61', '#FFC0CB']

    
    if isinstance(next(iter(curves_sd)), str):
        curves_sd = {1: curves_sd} 
    if isinstance(next(iter(curves_ista)), str):
        curves_ista = {1: curves_ista}

    fillvalue=(None, {'epochs': np.array([]), 'train': np.array([]), 'test': np.array([])})
    
    for (key1, value1), (key2, value2), color in zip_longest(curves_sd.items(), curves_ista.items(), colors, fillvalue=fillvalue):
        if len(value1['epochs']) > 0:
            adjusted_epochs_train = value1['epochs'] + current_epoch_train
            axs[0].plot(adjusted_epochs_train, value1['train'], color=color)
            current_epoch_train = adjusted_epochs_train[-1]

            if len(value1['test']) > 0: 
                adjusted_epochs_test = value1['epochs'] + current_epoch_test
                axs[1].plot(adjusted_epochs_test, value1['test'], color=color)
                current_epoch_test = adjusted_epochs_test[-1]

        if len(value2['epochs']) > 0: 
            adjusted_epochs_train = value2['epochs'] + current_epoch_train
            axs[0].plot(adjusted_epochs_train, value2['train'], linestyle='dashed', color=color)
            current_epoch_train = adjusted_epochs_train[-1]

            if len(value2['test']) > 0: 
                adjusted_epochs_test = value2['epochs'] + current_epoch_test
                axs[1].plot(adjusted_epochs_test, value2['test'], linestyle='dashed', color=color)
                current_epoch_test = adjusted_epochs_test[-1] 

    axs[1].set_xlabel('Epochs')
    axs[0].set_ylabel('MSE Loss')
    axs[1].set_ylabel('MSE Loss')
    axs[0].legend(['SD', 'ISTA'])
    axs[1].legend(['SD', 'ISTA'])

    axs[0].set_title('Training loss')
    axs[1].set_title('Test loss')

    for ax in axs.flat:
        ax.spines[['right', 'top']].set_visible(False)
    
    plt.show()
