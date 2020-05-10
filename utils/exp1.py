import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
from .utils import *
from PIL import Image as PILImage
import scipy.ndimage
from .pytorch_utils import device

import math
import sys

softmax = None
model = None

def plot_gan_training(losses, title, ax):
    n_itr = len(losses)
    xs = np.arange(n_itr)

    ax.plot(xs, losses, label='loss')
    ax.legend()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Loss')

def plot_dicriminator_heatmap(d, fig, ax, resolution=60):
    grid = np.zeros((resolution, resolution, 2))
    grid[:, :, 0] = np.linspace(-1, 1, resolution).reshape((1, -1))
    grid[:, :, 1] = np.linspace(1, -1, resolution).reshape((-1, 1))
    flat_grid = grid.reshape((-1, 2))
    d.eval()
    flat_grid = torch.Tensor(flat_grid).to(device).float()
    result = torch.Tensor.cpu(d.forward(flat_grid)).detach().numpy()
    y, x = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    z = result.reshape(resolution, resolution)
    z_min, z_max = 0, 1
    col = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('discriminator probability heatmap')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(col, ax=ax)

def experiment_gan_plot(data, samples, title, ax, is_spiral=False):
    if is_spiral:
        ax.scatter(data[:, 0], data[:, 1], label='real')
        ax.scatter(samples[:, 0], samples[:, 1], label='fake')
    else:
        ax.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
        ax.hist(data, bins=50, density=True, alpha=0.7, label='real')
    ax.legend()
    ax.grid()
    ax.set_title(title)

def experiment_data(n=20000, is_spiral=False, n_modes=1, params=[(0,1)]):
    if is_spiral:
        theta = np.random.rand(n) * 30 
        r = theta / 30
        results =np.zeros((n, 2))
        results[:, 0] = r * (np.cos(theta) + np.random.rand(n) / 7)
        results[:, 1] = r * (np.sin(theta) + np.random.rand(n) / 7)
        return results
    else:
        assert n % 2 == 0
        assert n_modes == len(params)
        lst = []
        for i in range(n_modes):
            gaussian = np.random.normal(loc=params[i][0], scale=params[i][1], size=(n//n_modes,))
            lst.append(gaussian)
        data = (np.concatenate(lst) + 1).reshape([-1, 1])
        scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        return 2 * scaled_data -1

def visualize_experiment_dataset(is_spiral=False, modes=1, param_modes=[(0,1)]):
    data = experiment_data(is_spiral=is_spiral, n_modes=modes, params=param_modes)
    plt.figure(figsize=(8,8))
    if is_spiral:
        plt.scatter(data[:, 0], data[:, 1], label='train spiral data')
    else:
        plt.hist(data, bins=50, alpha=0.7, label='train data')
    plt.legend()
    plt.show()


def experiment_save_results(part, fn, name, is_spiral=False, modes=1, param_modes=[(0,1)]):
    data = experiment_data(is_spiral=is_spiral, n_modes=modes, params=param_modes)
    g, c, losses, samples_start, samples_end, pvals = fn(data)
    fig = plt.figure(figsize=(12,30))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    plot_gan_training(losses, f'{name}{part} Losses', ax1)
    experiment_gan_plot(data,  samples_start, f'{name}{part} Epoch 1', ax2, is_spiral)
    experiment_gan_plot(data, samples_end, f'{name}{part} Final', ax3, is_spiral)
    # plt.show()
    savefig(f'results/{name}{part}.png')
    return g, c, losses, samples_start, samples_end, pvals


