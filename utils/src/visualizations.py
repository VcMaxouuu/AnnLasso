import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from IPython.display import HTML


def draw_loss_curves(curves_sd, curves_ista):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=500, sharex=True) 
    current_epoch = 0

    colors = ['#5A5B9F', '#D94F70', '#009473', '#F0C05A', '#7BC4C4', '#FF6F61', '#FFC0CB']

    
    if isinstance(next(iter(curves_sd)), str):
        curves_sd = {1: curves_sd} 
    if isinstance(next(iter(curves_ista)), str):
        curves_ista = {1: curves_ista}

    fillvalue=(None, {'epochs': np.array([]), 'cost': np.array([]), 'train': np.array([])})
    
    for (key1, value1), (key2, value2), color in zip_longest(curves_sd.items(), curves_ista.items(), colors, fillvalue=fillvalue):
        if len(value1['epochs']) > 0:
            adjusted_epochs = value1['epochs'] + current_epoch
            axs[0].plot(adjusted_epochs, value1['cost'], color=color)
            axs[1].plot(adjusted_epochs, value1['train'], color=color)
            current_epoch= adjusted_epochs[-1]

        if len(value2['epochs']) > 0: 
            adjusted_epochs = value2['epochs'] + current_epoch
            axs[0].plot(adjusted_epochs, value2['cost'], linestyle='dashed', color=color)
            axs[1].plot(adjusted_epochs, value2['train'], linestyle='dashed', color=color)
            current_epoch= adjusted_epochs[-1]
    

    axs[1].set_xlabel('Epochs')
    axs[0].set_ylabel('Cost')
    axs[1].set_ylabel('MSE Loss')
    for ax in axs.flat:
        custom_lines = [Line2D([0], [0], color='black', lw=1),
                Line2D([0], [0], color='black', lw=1, linestyle='dashed')]
        ax.legend(custom_lines, ['SD', 'ISTA'])

    axs[0].set_title('Cost function')
    axs[1].set_title('Training MSE')

    for ax in axs.flat:
        ax.spines[['right', 'top']].set_visible(False)
    
    plt.show()


def draw_layer1_evolution(layer1_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax1.set_title("Weights")
    ax2.set_title("Biases")

    if isinstance(next(iter(layer1_history)), str):
        layer1_history = {1: layer1_history}
    
    all_weights, all_bias, lambdas = [], [], []

    for key, value in layer1_history.items():
        all_weights.extend(value['weight'])
        all_bias.extend(value['bias'])
        lambdas.append(key)

    weights = np.abs(np.array(all_weights))
    bias = np.abs(np.array(all_bias)) 
    weights /= weights.max()
    bias /= bias.max()

    lim_weights = weights[0].shape
    lim_bias = bias[0].shape

    im1 = ax1.imshow(weights[0], aspect=lim_weights[1]/lim_weights[0], cmap='plasma', extent=[0,lim_weights[1],0,lim_weights[0]])
    im2 = ax2.imshow(np.expand_dims(bias[0], axis=1), cmap='plasma', extent=[0, 1, 0, lim_bias[0]])

    ax1.grid(True)
    ax2.grid(True)

    ax1.set_yticks(np.arange(0, lim_weights[0], 2))
    ax2.set_xticks([])

    def init():
        im1.set_data(weights[0])
        im2.set_data(np.expand_dims(bias[0], axis=1))
        return (im1, im2)

    def update(i):
        im1.set_data(weights[i])
        im2.set_data(np.expand_dims(bias[i], axis=1))
        return (im1, im2)

    anim = FuncAnimation(fig, update, frames=range(len(weights)), init_func=init, blit=True, cache_frame_data=False)

    def is_running_in_notebook():
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except Exception:
            return False
    
        return True

    if is_running_in_notebook():
        return HTML(anim.to_jshtml())
    else:
        return plt.show()
