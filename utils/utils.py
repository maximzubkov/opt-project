import os
from os.path import dirname, exists

import matplotlib.pyplot as plt
import numpy as np


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label="train loss", linewidth=7.0)
    plt.plot(x_test, test_losses, label="test loss", linewidth=7.0)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    savefig(fname)


def compare_pvals(path="results/compare_sn_tmp.pdf"):
    plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ax1.set_title("SNGAN p-value", fontsize=45)
    ax2.set_title("SNGAN Generator grad norm", fontsize=45)
    for n_cr in [2, 4, 6]:
        ax1.semilogy(np.load(f"results/sn_ncr{n_cr}/p.npy"), label="$n_{cr}= $" + f" {n_cr}", linewidth=7.0)
        ax1.set_xlabel("Training Iteration", fontsize=42)
        ax1.set_ylabel("p-value", fontsize=42)
        ax1.tick_params(axis="x", labelsize=40)
        ax1.tick_params(axis="y", labelsize=40)
        ax2.semilogy(np.load(f"results/sn_ncr{n_cr}/g_grad.npy"), label="$n_{cr}= $" + f" {n_cr}", linewidth=7.0)
        ax2.set_xlabel("Training Iteration", fontsize=42)
        ax2.set_ylabel("Generator Grad Norm", fontsize=42)
        ax2.tick_params(axis="x", labelsize=40)
        ax2.tick_params(axis="y", labelsize=40)
    ax1.grid()
    ax1.legend(fontsize=30)
    ax2.grid()
    ax2.legend(fontsize=30)

    savefig(path)


def experiment_save_results(experiment):
    g, c, losses, samples_start, samples_end = experiment.run()
    experiment.final_plot(g, c, losses, samples_start, samples_end)
    return g, c, losses


def sn2nosn(path="results/compare_tmp.pdf"):
    plt.figure(figsize=(52, 55))
    ax = {}
    ax[2] = {}
    ax[4] = {}
    ax[6] = {}
    ax[2][0] = plt.subplot(3, 3, 1)
    ax[2][1] = plt.subplot(3, 3, 2)
    ax[2][2] = plt.subplot(3, 3, 3)
    ax[4][0] = plt.subplot(3, 3, 4)
    ax[4][1] = plt.subplot(3, 3, 5)
    ax[4][2] = plt.subplot(3, 3, 6)
    ax[6][0] = plt.subplot(3, 3, 7)
    ax[6][1] = plt.subplot(3, 3, 8)
    ax[6][2] = plt.subplot(3, 3, 9)
    for n_cr in [2, 4, 6]:
        ax[n_cr][0].set_title("$n_{cr}= $" + f" {n_cr}", fontsize=65)
        ax[n_cr][0].semilogy(np.load(f"results/sn_ncr{n_cr}/p.npy"), label=f"SNGAN", linewidth=7.0)
        ax[n_cr][0].semilogy(np.load(f"results/ncr{n_cr}/p.npy"), label=f"noSNGAN", linewidth=7.0)
        ax[n_cr][0].set_xlabel("Training Iteration", fontsize=55)
        ax[n_cr][0].set_ylabel("p-value", fontsize=55)
        ax[n_cr][0].tick_params(axis="x", labelsize=50)
        ax[n_cr][0].tick_params(axis="y", labelsize=50)
        ax[n_cr][0].grid()
        ax[n_cr][0].legend(fontsize=30)
        x = [300 * x for x in range(len(np.load(f"results/sn_ncr{n_cr}/g_grad.npy")))]
        ax[n_cr][1].set_title("$n_{cr}= $" + f" {n_cr}", fontsize=65)
        ax[n_cr][1].semilogy(x, np.load(f"results/sn_ncr{n_cr}/g_grad.npy"), label=f"SNGAN", linewidth=7.0)
        x = [300 * x for x in range(len(np.load(f"results/ncr{n_cr}/g_grad.npy")))]
        ax[n_cr][1].semilogy(x, np.load(f"results/ncr{n_cr}/g_grad.npy"), label=f"noSNGAN", linewidth=7.0)
        ax[n_cr][1].set_xlabel("Training Iteration", fontsize=55)
        ax[n_cr][1].set_ylabel("Generator Grad Norm", fontsize=55)
        ax[n_cr][1].tick_params(axis="x", labelsize=50)
        ax[n_cr][1].tick_params(axis="y", labelsize=50)
        ax[n_cr][1].grid()
        ax[n_cr][1].legend(fontsize=30)
        ax[n_cr][2].set_title("$n_{cr}= $" + f" {n_cr}", fontsize=65)
        ax[n_cr][2].plot(np.load(f"results/sn_ncr{n_cr}/accuracy.npy"), label=f"SNGAN", linewidth=7.0)
        ax[n_cr][2].plot(np.load(f"results/ncr{n_cr}/accuracy.npy"), label=f"noSNGAN", linewidth=7.0)
        ax[n_cr][2].set_xlabel("Training Iteration", fontsize=55)
        ax[n_cr][2].set_ylabel("Discriminator accuracy", fontsize=55)
        ax[n_cr][2].tick_params(axis="x", labelsize=50)
        ax[n_cr][2].tick_params(axis="y", labelsize=70)
        ax[n_cr][2].grid()
        ax[n_cr][2].legend(fontsize=30)

    savefig(path)
