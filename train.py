import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm_notebook
import utils.pytorch_utils as ptu
from utils.exp1 import experiment_gan_plot, experiment_data
import numpy as np
from IPython.display import clear_output
import scipy.stats as stat


def train(generator, critic, c_loss_fn, g_loss_fn, 
          train_loader, g_optimizer, c_optimizer, gp_lamb=0.001,
          n_critic=1, g_scheduler=None, c_scheduler=None, 
          weight_clipping=None):
    g_losses, c_losses = [], []
    generator.train()
    critic.train()
    for i, x in enumerate(train_loader):
        x = x.to(ptu.device).float()
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
    return dict(g_losses=g_losses, c_losses=c_losses)

def train_epochs(generator, critic, g_loss_fn, c_loss_fn, 
                 train_loader, train_args, g_opt, c_opt, 
                 g_scheduler=None, c_scheduler=None, is_spiral=False, modes=1, param_modes=[(0,1)]):
    epochs = train_args['epochs']

    train_losses = dict()
    data = experiment_data(is_spiral=is_spiral, n_modes=modes, params=param_modes)
    pvals = []

    for epoch in tqdm_notebook(range(epochs), desc='Epoch', leave=False):
        if epoch == 0:
            start_snapshot = get_training_snapshot(generator, critic)
        generator.train()
        critic.train()
        train_loss = train(generator, critic, c_loss_fn, g_loss_fn, train_loader, 
                           g_opt, c_opt, gp_lamb=train_args.get('n_critic', 0.001),
                           n_critic=train_args.get('n_critic', 1), 
                           g_scheduler=g_scheduler, c_scheduler=c_scheduler)
        
        for k in train_loss:
            if k not in train_losses:
                train_losses[k] = []
            train_losses[k].extend(train_loss[k])
        sample = get_training_snapshot(generator, critic)

        # stat criterion
        data2 = np.array(sample)
        data2 = data2.T
        data1 = data.T
        pvalue = stat.ks_2samp(data1[0], data2[0])[1]
        pvals.append(pvalue)

        experiment_gan_plot(data, sample, f'Epoch {epoch}',
                            f'results/epoch_{epoch}.png', is_spiral)
        clear_output(wait=True)
    if train_args.get('final_snapshot', False):
        final_snapshot = get_training_snapshot(generator, critic)
        return (train_losses, start_snapshot, final_snapshot, pvals)
    else:
        return train_losses, pvals

def get_training_snapshot(generator, critic, n_samples=5000):
    generator.eval()
    critic.eval()
    samples = ptu.get_numpy(generator.sample(n_samples))
    return samples
