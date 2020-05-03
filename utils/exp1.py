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

def experiment_gan_plot(data, samples, xs, ys, title, fname):
    plt.figure()
    plt.hist(samples, bins=50, density=True, alpha=0.7, label='fake')
    plt.hist(data, bins=50, density=True, alpha=0.7, label='real')

    plt.plot(xs, ys, label='discrim')
    plt.legend()
    plt.title(title)
    savefig(fname)
    
def spiral_experiment_gan_plot(data, samples, xs, title, fname):
    plt.figure(figsize=(8,8))
    plt.scatter(data[:, 0], data[:, 1], label='real')
    plt.scatter(samples[:, 0], samples[:, 1], label='fake')
    plt.grid()

    plt.legend()
    plt.title(title)
    savefig(fname)

def experiment_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n//2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n//2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data -1

def spiral_experiment_data(n_samples=20000):
    theta = np.random.rand(n_samples) * 30 
    r = theta / 30
    results =np.zeros((n_samples, 2))
    results[:, 0] = r * (np.cos(theta) + np.random.rand(n_samples) / 7)
    results[:, 1] = r * (np.sin(theta) + np.random.rand(n_samples) / 7)
    return results

def visualize_experiment_dataset():
    data = experiment_data()
    plt.hist(data, bins=50, alpha=0.7, label='train data')
    plt.legend()
    plt.show()
    
def visualize_spiral_experiment_dataset():
    data = spiral_experiment_data()
    plt.figure(figsize=(8,8))
    plt.scatter(data[:, 0], data[:, 1], label='train spiral data')
    plt.grid()

def experiment_save_results(part, fn, name):
    data = experiment_data()
    losses, samples1_start, xs1_start, ys1_start, samples_end, xs_end, ys_end = fn(data)
    plot_gan_training(losses, f'{name}{part} Losses', f'results/{name}{part}_losses.png')
    experiment_gan_plot(data,  samples_start, xs_start, ys_start, f'{name}{part} Epoch 1', f'results/{name}{part}_epoch1.png')
    experiment_gan_plot(data, samples_end, xs_end, ys_end, f'{name}{part} Final', f'results/{name}{part}_final.png')
    
def spiral_experiment_save_results(part, fn, name="spiral"):
    data = spiral_experiment_data()
    losses, samples_start, xs_start, ys_start, samples_end, xs_end, ys_end = fn(data)
    plot_gan_training(losses, f'{name}{part} Losses', f'results/{name}{part}_losses.png')
    spiral_experiment_gan_plot(data, samples_start, xs_start, f'{name}{part} Epoch 1', f'results/{name}{part}_epoch1.png')
    spiral_experiment_gan_plot(data, samples_end, xs_end, f'{name}{part} Final', f'results/{name}{part}_final.png')


