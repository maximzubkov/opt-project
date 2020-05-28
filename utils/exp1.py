import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as transforms
from .utils import *
from PIL import Image as PILImage
import scipy.ndimage
from .pytorch_utils import device
from IPython.display import clear_output
import statsmodels.api as sm
from scipy import stats, integrate
from statsmodels.graphics.gofplots import qqplot_2samples
from sklearn.neighbors import KernelDensity

import math
import sys

softmax = None
model = None

def kde(X, X_grid, bandwidth=0.2):
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(X)
    log_pdf = kde_skl.score_samples(X_grid[:, None])
    return np.exp(log_pdf)

def plot_gan_training(losses, title, ax):
    n_itr = len(losses)
    xs = np.arange(n_itr)

    ax.plot(xs, losses, label='loss')
    ax.set_ylim([-0.1, 0.1])
    ax.legend(prop={'size': 40})
    ax.grid()
    ax.set_title(title, fontsize=45)
    ax.set_xlabel('Training Iteration',fontsize=42)
    ax.set_ylabel('Loss',fontsize=42)
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    

def plot_dicriminator_heatmap(d, fig, ax, resolution=60):
    grid = np.zeros((resolution, resolution, 2))
    grid[:, :, 0] = np.linspace(-1, 1, resolution).reshape((1, -1))
    grid[:, :, 1] = np.linspace(1, -1, resolution).reshape((-1, 1))
    flat_grid = grid.reshape((-1, 2))
    d.eval()
    flat_grid = torch.Tensor(flat_grid).to(device).float()
    result = torch.Tensor.cpu(d.forward(flat_grid)).detach().numpy()
    x, y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    z = result.reshape(resolution, resolution)
    z_min, z_max = z.min(), z.max()
    col = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=0, vmax=1)
    ax.set_title('discriminator probability heatmap')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(col, ax=ax)

def experiment_gan_plot(data, samples, title, ax, is_spiral=False):
    if is_spiral:
        ax.scatter(data[:, 0], data[:, 1], label='real')
        ax.scatter(samples[:, 0], samples[:, 1], label='fake')
    else:
        data_grid = np.linspace(data.min(), data.max(), 1000)
        sample_grid = np.linspace(samples.min(), samples.max(), 1000)
        bandwidth = 0.1
        data_p_n = kde(data, data_grid, bandwidth=bandwidth)
        sample_p_n = kde(samples, sample_grid, bandwidth=bandwidth)
        ax.fill_between(x=data_grid, y1=data_p_n, y2=0, alpha=0.7, label='real')
        ax.fill_between(x=sample_grid, y1=sample_p_n, y2=0, alpha=0.7, label='fake')
    ax.legend(prop={'size': 40})
    ax.grid()
    ax.set_title(title, fontsize=45)
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
    
def show_qq_plot(data, current, previous, title, ax, is_spiral=False):
    pp_x = sm.ProbPlot(current)
    pp_y = sm.ProbPlot(previous)
    qqplot_2samples(pp_x, pp_y, line="r", ax=ax)
    ax.grid()
    ax.set_title(title)

def experiment_data(n=50000, is_spiral=False, n_modes=1, params=[(0,1)]):
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
    plt.figure(figsize=(20,20))
    plt.rc("text", usetex=True)
    plt.title("Initial Distribution", fontsize=45)
    if is_spiral:
        plt.scatter(data[:, 0], data[:, 1], label='train spiral data')
    else:
        data_grid = np.linspace(data.min(), data.max(), 1000)
        bandwidth = 0.1
        data_p_n = kde(data, data_grid, bandwidth=bandwidth)
        plt.fill_between(x=data_grid, y1=data_p_n, y2=0, alpha=0.7)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid()
    savefig("results/intial_distr.pdf")
    plt.show()

def experiment_save_results(part, fn, name, is_spiral=False, modes=1, param_modes=[(0,1)]):
    data = experiment_data(is_spiral=is_spiral, n_modes=modes, params=param_modes)
    g, c, losses, samples_start, samples_end = fn(data)
    clear_output(wait=True)
    fig = plt.figure(figsize=(30,30))
    ax1 = fig.add_subplot(3, 2, 3)
    ax2 = fig.add_subplot(3, 2, 4)
    ax3 = fig.add_subplot(3, 2, 1)
    ax4 = fig.add_subplot(3, 2, 2)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)
    ax1.semilogy(losses["pvals"])
    ax1.set_title("p-value", fontsize=45)
    ax1.set_xlabel('Epoch', fontsize=42)
    ax1.set_ylabel('p-value', fontsize=42)
    ax1.tick_params(axis="x", labelsize=40)
    ax1.tick_params(axis="y", labelsize=40)
    ax1.grid()
    
    ax2.plot(losses["accuracy"])
    ax2.set_title("Accuracy", fontsize=45)
    ax2.set_xlabel('Epoch', fontsize=42)
    ax2.set_ylabel('Accuracy', fontsize=42)
    ax2.tick_params(axis="x", labelsize=40)
    ax2.tick_params(axis="y", labelsize=40)
    ax2.grid()
    x = [600 * x for x in range(len(losses["g_grad"]))]
    ax5.semilogy(x, losses["g_grad"])
    ax5.set_title("Norm of Generator gradient", fontsize=45)
    ax5.set_xlabel('Training Iteration',fontsize=42)
    ax5.set_ylabel('Norm of gradient',fontsize=42)
    ax5.tick_params(axis="x", labelsize=40)
    ax5.tick_params(axis="y", labelsize=40)
    ax5.grid()
    x = [600 * x for x in range(len(losses["c_grad"]))]
    ax6.semilogy(x, losses["c_grad"])
    ax6.set_title("Norm of Discriminator gradient", fontsize=45)
    ax6.set_xlabel('Training Iteration',fontsize=42)
    ax6.set_ylabel('Norm of gradient',fontsize=42)
    ax6.tick_params(axis="x", labelsize=40)
    ax6.tick_params(axis="y", labelsize=40)
    ax6.grid()

    experiment_gan_plot(data,  samples_start, f'Epoch 1', ax3, is_spiral)
    experiment_gan_plot(data, samples_end, f'Epoch 50', ax4, is_spiral)
    savefig(f'results/{name}/{name}.pdf')
    plt.show()
    return g, c, losses, samples_start, samples_end


