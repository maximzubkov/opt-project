import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm_notebook
import utils.pytorch_utils as ptu
from utils.exp1 import experiment_gan_plot, experiment_data, plot_dicriminator_heatmap, show_qq_plot
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import scipy.stats as stat
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples


def train(generator, critic, c_loss_fn, g_loss_fn, 
          train_loader, g_optimizer, c_optimizer, gp_lamb=0.001,
          n_critic=1, g_scheduler=None, c_scheduler=None, 
          weight_clipping=None):
    g_losses, c_losses, accuracy = [], [], []
    generator.train(True)
    critic.train(True)
    for i, x in enumerate(train_loader):
        x = x.to(ptu.device).float()
        if i % 10 == 0:
            generator.train(False)
            critic.train(False)
            sample = generator.sample(x.shape[0])
            preds1 = critic(sample).detach().numpy()
            preds2 = critic(x).detach().numpy()
            predictions_false = preds1 <= 0.5
            predictions_true = preds2 > 0.5
            correct = np.sum(predictions_false) + np.sum(predictions_true)
            acc = correct / (x.shape[0] * 2)
            accuracy.append(acc)
            generator.train(True)
            critic.train(True)
        fake_data = generator.sample(x.shape[0])

        c_loss = c_loss_fn(generator, critic, x)
        c_optimizer.zero_grad()
        c_loss.backward()
        c_optimizer.step()
        c_losses.append(c_loss.item())

        if i % n_critic == 0:
            g_loss = g_loss_fn(generator, critic, x)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            g_losses.append(g_loss.item())
            if g_scheduler is not None:
                g_scheduler.step()
            if c_scheduler is not None:
                c_scheduler.step()
    return dict(g_losses=g_losses, c_losses=c_losses, accuracy=accuracy)

def train_epochs(generator, critic, g_loss_fn, c_loss_fn, 
                 train_loader, train_args, g_opt, c_opt, 
                 g_scheduler=None, c_scheduler=None, is_spiral=False, modes=1, param_modes=[(0,1)]):
    epochs = train_args['epochs']

    train_losses = dict()
    data = experiment_data(is_spiral=is_spiral, n_modes=modes, params=param_modes)
    pvals = []
    snapshots = []

    for epoch in tqdm_notebook(range(epochs), desc='Epoch', leave=False):
        if epoch == 0:
            start_snapshot = get_training_snapshot(generator, critic)
        generator.train(True)
        critic.train(True)
        train_loss = train(generator, critic, c_loss_fn, g_loss_fn, train_loader, 
                           g_opt, c_opt, gp_lamb=train_args.get('n_critic', 0.001),
                           n_critic=train_args.get('n_critic', 1), 
                           g_scheduler=g_scheduler, c_scheduler=c_scheduler)
        
        for k in train_loss:
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].extend(train_loss[k])
        sample = get_training_snapshot(generator, critic)
        snapshots.append(sample)

        if not is_spiral:
            # stat criterion
            data2 = np.array(sample)
            data2 = data2.T
            data1 = data.T
            pvalue = stat.ks_2samp(data1[0], data2[0])[1]
            pvals.append(pvalue)

        clear_output(wait=True)
        if is_spiral:
            fig = plt.figure(figsize=(20,10))

            ax1 = fig.add_subplot(3, 1, 1)
            ax2 = fig.add_subplot(3, 1, 2)
            ax3 = fig.add_subplot(3, 1, 3)
            # ax2 = fig.add_subplot(1, 2, 2)
            # ax3 = fig.add_subplot(2, 2, 3)
            # ax4 = fig.add_subplot(2, 2, 4)
            
            experiment_gan_plot(data, sample, f'Epoch {epoch}', ax=ax1, is_spiral=True)
            # show_qq_plot(data, sample, snapshots[epoch-1], f'Q-Q curr_prev Epoch {epoch}', ax=ax3, is_spiral=True)
            # show_qq_plot(data, sample, data, f'Q-Q curr_target Epoch {epoch}', ax=ax4, is_spiral=True)
            # plot_dicriminator_heatmap(critic, fig=fig, ax=ax2)

            ax2.plot(train_losses["g_losses"])
            ax2.set_title("g loss")
            ax2.grid()
            ax3.plot(train_losses["c_losses"])
            ax3.set_title("d loss")
            ax3.grid()
            plt.show()
        else:
            fig = plt.figure(figsize=(20,20))

            ax1 = fig.add_subplot(4, 1, 1)
            ax2 = fig.add_subplot(4, 1, 2)
            ax3 = fig.add_subplot(4, 1, 3)
            ax4 = fig.add_subplot(4, 1, 4)
            experiment_gan_plot(data, sample, f'Epoch {epoch}', ax=ax1, is_spiral=False)
            ax2.plot(train_losses["g_losses"])
            ax2.set_title("g loss")
            ax2.grid()
            ax3.plot(train_losses["c_losses"])
            ax3.set_title("d loss")
            ax3.grid()
            ax4.plot(train_losses["accuracy"])
            ax4.set_title("acc")
            ax4.grid()
            # show_qq_plot(data, sample, data, f'Q-Q curr_target Epoch {epoch}', ax=ax2, is_spiral=False)
            plt.show()
        
    if train_args.get('final_snapshot', False):
        final_snapshot = get_training_snapshot(generator, critic)
        return (train_losses, start_snapshot, final_snapshot, pvals)
    else:
        return train_losses, pvals

def get_training_snapshot(generator, critic, n_samples=10000):
    generator.train(False)
    critic.train(False)
    samples = ptu.get_numpy(generator.sample(n_samples))
    generator.train(True)
    critic.train(True)
    return samples
