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
    g_losses, g_grad, c_grad, c_losses, accuracy = [], [], [], [], []
    generator.train(True)
    critic.train(True)
    for i, x in enumerate(train_loader):
        x = x.to(ptu.device).float()
        fake_data = generator.sample(x.shape[0])

        c_loss = c_loss_fn(generator, critic, x)
        c_optimizer.zero_grad()
        c_loss.backward()
        if i % 300 == 0:
            c_grad.append(0)
            for param in critic.parameters():
                if param.requires_grad:
                    c_grad[-1] += torch.norm(param.grad.data).detach().numpy() ** 2
        c_optimizer.step()
        # if i % 1000 == 0:
        #     c_losses.append(c_loss.item())

        if i % n_critic == 0:
            g_loss = g_loss_fn(generator, critic, x)
            g_optimizer.zero_grad()
            g_loss.backward()
            if i % 300 == 0:
                g_grad.append(0)
                for param in generator.parameters():
                    if param.requires_grad:
                        g_grad[-1] += torch.norm(param.grad.data).detach().numpy() ** 2
            g_optimizer.step()
            # if i % 1000 == 0:
            #     g_losses.append(g_loss.item())
            if g_scheduler is not None:
                g_scheduler.step()
            if c_scheduler is not None:
                c_scheduler.step()
    return dict(g_losses=g_losses, c_losses=c_losses, accuracy=accuracy, g_grad=g_grad, c_grad=c_grad)

def train_epochs(generator, critic, g_loss_fn, c_loss_fn,
                 train_loader, train_args, g_opt, c_opt, 
                 g_scheduler=None, c_scheduler=None, is_spiral=False, modes=1, param_modes=[(0,1)], name=""):
    epochs = train_args['epochs']

    train_losses = dict()
    train_losses["pvals"] = []
    train_losses["accuracy"] = []
    train_losses["hist_diff"] = []
    data = experiment_data(is_spiral=is_spiral, n_modes=modes, params=param_modes, n=8192)

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


        if not is_spiral:
            # stat criterion
            generator.train(False)
            critic.train(False)
            data1 = torch.Tensor(experiment_data(n=250, is_spiral=is_spiral, n_modes=modes, params=param_modes))
            data2 = generator.sample(250)
            preds1 = critic(data2).detach().numpy()
            preds2 = critic(data1).detach().numpy()
            predictions_false = preds1 <= 0.5
            predictions_true = preds2 > 0.5
            correct = np.sum(predictions_false) + np.sum(predictions_true)
            acc = correct / (data1.shape[0] * 2)
            train_losses["accuracy"].append(acc)
            generator.train(True)
            critic.train(True)
            data1 = data1.detach().numpy()
            data2 = data2.detach().numpy()
            pvalue = stat.ks_2samp(data1.T[0], data2.T[0])[1]
            train_losses["pvals"].append(pvalue)

        # clear_output(wait=True)
        if is_spiral:
            fig = plt.figure(figsize=(20,10))
            plt.rc("text", usetex=True)

            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            
            experiment_gan_plot(data, sample, f'Epoch {epoch}', ax=ax1, is_spiral=True)
            plot_dicriminator_heatmap(critic, fig=fig, ax=ax2)
            plt.show()
        else:
            # fig = plt.figure(figsize=(20,20))
            # plt.rc("text", usetex=True)
            # ax = fig.add_subplot(1, 1, 1)
            # experiment_gan_plot(data, data2, f'Epoch {epoch}', ax=ax, is_spiral=False)
            # plt.savefig(f"results/{name}/hist_{epoch}.pdf") 
            # plt.show()
            if epoch == epochs - 1:
                fig = plt.figure(figsize=(20,20))
                plt.rc("text", usetex=True)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
                ax1.plot(train_losses["g_grad"])
                ax1.set_title("Norm of Generator gradient", fontsize=45)
                ax1.set_xlabel('Training Iteration',fontsize=42)
                ax1.set_ylabel('Norm of gradient',fontsize=42)
                ax1.tick_params(axis="x", labelsize=40)
                ax1.tick_params(axis="y", labelsize=40)
                ax1.grid()
                
                ax2.plot(train_losses["c_grad"])
                ax2.set_title("Norm of Discriminator gradient", fontsize=45)
                ax2.set_xlabel('Training Iteration',fontsize=42)
                ax2.set_ylabel('Norm of gradient',fontsize=42)
                ax2.tick_params(axis="x", labelsize=40)
                ax2.tick_params(axis="y", labelsize=40)
                ax2.grid()

                plt.savefig(f"results/{name}/output_{epoch}.pdf") 
            plt.show()
        
    if train_args.get('final_snapshot', False):
        final_snapshot = get_training_snapshot(generator, critic)
        return (train_losses, start_snapshot, final_snapshot)
    else:
        return train_losses

def get_training_snapshot(generator, critic, n_samples=10000):
    generator.train(False)
    critic.train(False)
    samples = ptu.get_numpy(generator.sample(n_samples))
    generator.train(True)
    critic.train(True)
    return samples
