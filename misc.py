import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve


def plot_1component(x, y_fit, y=False, output_path=False, thermostat='NpT', title="Fitted force function",
                    labels=("true", "fit"), y_label=r"$f_{ij}$"):
    fig, ax = plt.subplots(1, 1)
    if y is not False:
        ax.plot(x, y, label=f"{labels[0]}", lw=2.5, color='xkcd:azure')
    ax.axhline(0, ls='--', color='xkcd:light grey')
    ax.plot(x, y_fit, label=f"{labels[1]}", ls='-.', lw=2., color='xkcd:bright orange')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{title} ({thermostat})")
    ax.set_xlabel(r"$r_{ij}$", fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.legend(frameon=False)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')


def plot_2component(X, Y_fit, Y=np.empty(0), output_path=False, thermostat='NpT'):
    fig, ax = plt.subplots(3, 1, figsize=(5, 15))
    i = 0
    types = ['11', '12', '22']
    for force in [0, 1, 2]:
        if len(Y) > 0:
            ax[i].plot(X[force], Y[force], label='true', lw=2.5, color='xkcd:azure')
        ax[i].axhline(0, ls='--', color='xkcd:light grey')
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


def lowess(y, f=0.01, iter=1):
    x = np.arange(0, len(y), 1)
    n = len(y)
    r = int(math.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


def radial_distribution_function(x, y, z, s, r_max, dr):
    """via https://github.com/cfinch/Shocksolution_Examples/blob/master/PairCorrelation/paircorrelation.py"""

    bools1 = x > r_max
    bools2 = x < (s - r_max)
    bools3 = y > r_max
    bools4 = y < (s - r_max)
    bools5 = z > r_max
    bools6 = z < (s - r_max)

    interior_indices, = np.where(bools1 * bools2 * bools3 * bools4 * bools5 * bools6)
    num_interior_particles = len(interior_indices)
    if num_interior_particles < 1:
        raise RuntimeError("No particles found for which a sphere of radius r_max\
                will lie entirely within a cube of side length S.  Decrease r_max\
                or increase the size of the cube.")

    edges = np.arange(0., r_max + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = np.zeros([num_interior_particles, num_increments])
    radii = np.zeros(num_increments)
    number_density = len(x) / s ** 3

    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = np.sqrt((x[index] - x) ** 2 + (y[index] - y) ** 2 + (z[index] - z) ** 2)
        d[index] = 2 * r_max

        (result, bins) = np.histogram(d, bins=edges, normed=False)
        g[p, :] = result / number_density

    g_average = np.zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i + 1]) / 2.
        r_outer = edges[i + 1]
        r_inner = edges[i]
        g_average[i] = np.mean(g[:, i]) / (4.0 / 3.0 * np.pi * (r_outer ** 3 - r_inner ** 3))

    return g_average, radii, interior_indices


def mean_squared_displacement():
    msd = []
    return msd



def augment(x, y, z, s):
    x, y, z = list(x), list(y), list(z)

    x_pos = list(np.array(x) + s)
    x_neg = list(np.array(x) - s)
    x = x_neg + x + x_pos
    y = y * 3
    z = z * 3

    y_pos = list(np.array(y) + s)
    y_neg = list(np.array(y) - s)
    y = y_neg + y + y_pos
    x = x * 3
    z = z * 3

    z_pos = list(np.array(z) + s)
    z_neg = list(np.array(z) - s)
    z = z_neg + z + z_pos
    x = x * 3
    y = y * 3

    return np.array(x), np.array(y), np.array(z)
