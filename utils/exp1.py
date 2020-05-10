import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
from .utils import *
from PIL import Image as PILImage
import scipy.ndimage
import cv2

import numpy as np
import math
import sys

softmax = None
model = None
device = torch.device("cuda:0")

def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)

def experiment_gan_plot(data, samples, title, fname, is_spiral=False):
    plt.figure(figsize=(8,8))
    if is_spiral:
        plt.scatter(data[:, 0], data[:, 1], label='real')
        plt.scatter(samples[:, 0], samples[:, 1], label='fake')
    else:
        plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
        plt.hist(data, bins=50, density=True, alpha=0.7, label='real')
    plt.legend()
    plt.grid()
    plt.title(title)
    savefig(fname)

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
    plot_gan_training(losses, f'{name}{part} Losses', f'results/{name}{part}_losses.png')
    experiment_gan_plot(data,  samples_start, f'{name}{part} Epoch 1', f'results/{name}{part}_epoch1.png', is_spiral)
    experiment_gan_plot(data, samples_end, f'{name}{part} Final', f'results/{name}{part}_final.png', is_spiral)
    return g, c, losses, samples_start, samples_end, pvals

