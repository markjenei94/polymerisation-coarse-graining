import numpy as np
import matplotlib.pyplot as plt


def plot_1component(x, y, y_fit, output_path=False, thermostat='NpT'):

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, label='true', lw=2.5, color='xkcd:azure')
    ax.plot(x, y_fit, label='fit', ls='-.', lw=2., color='xkcd:bright orange')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Fitted force function ({thermostat})")
    ax.set_xlabel(r"$r_{ij}$", fontsize=15)
    ax.set_ylabel(r"$f_{ij}$", fontsize=15)
    ax.legend(frameon=False)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')

def plot_2component(X, Y, Y_fit, output_path=False, thermostat='NpT'):
    fig, ax = plt.subplots(3, 1, figsize=(5, 15))
    i = 0
    types = ['11', '12', '22']
    for force in [0, 1, 2]:
        ax[i].plot(X[force], Y[force], label='true', lw=2.5, color='xkcd:azure')
        ax[i].plot(X[force], Y_fit[force], label='fit', ls='-.', lw=2., color='xkcd:bright orange')
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["bottom"].set_visible(False)
        ax[i].spines["left"].set_visible(False)
        if i == 0:
            ax[i].set_title(f"Fitted force function ({thermostat})")
        ax[i].set_xlabel(r"$r_{ij}$", fontsize=15)
        ax[i].set_ylabel(f"F({types[i]})", fontsize=13)
        ax[i].legend(frameon=False)
        i += 1

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')