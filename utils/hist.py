import scipy.stats as stat
import torch
import torch.optim as optim
import torch.utils.data as data
from IPython.display import clear_output
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler

from models import Generator, Discriminator, SN_Generator, SN_Discriminator
from train import train_epochs
from .pytorch_utils import device
from .utils import *

softmax = None
model = None


class HistogramExperiment:
    def __init__(self, config):
        self.config = config
        self.data = self.dataset()
        if self.config.sn:
            self.generator = SN_Generator
            self.discriminator = SN_Discriminator
        else:
            self.generator = Generator
            self.discriminator = Discriminator

    def dataset(self, n=50000):
        n_modes = len(self.config.dataset_params)
        gaussians = [np.random.normal(loc=loc, scale=scale, size=(n // n_modes,)) for loc, scale in
                     self.config.dataset_params]
        data = MinMaxScaler().fit_transform(np.concatenate(gaussians).reshape(-1, 1) + 1)
        return 2 * data - 1

    @staticmethod
    def _kde(x, x_grid, bandwidth=0.2):
        kde_skl = KernelDensity(bandwidth=bandwidth)
        kde_skl.fit(x)
        log_pdf = kde_skl.score_samples(x_grid[:, None])
        return np.exp(log_pdf)

    def run(self):
        '''
        Run trainiung of GAN with certain configuration
        '''
        loader_args = dict(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        train_loader = data.DataLoader(self.data, **loader_args)
        g = self.generator(*self.config.g_params).to(device)
        c = self.discriminator(*self.config.c_params).to(device)
        c_opt = optim.Adam(c.parameters(),
                           lr=self.config.c_lr,
                           betas=self.config.c_betas)
        g_opt = optim.Adam(g.parameters(),
                           lr=self.config.g_lr,
                           betas=self.config.g_betas)
        train_args = {
            "epochs": self.config.n_epochs,
            "n_critic": self.config.n_cr,
            "final_snapshot": True,
        }
        result = train_epochs(self, g, c, self.config.g_loss, self.config.c_loss,
                              train_loader, train_args,
                              g_opt=g_opt, c_opt=c_opt,
                              name=self.config.exp_name)

        train_losses, samples_start, samples_end = result
        return g, c, train_losses, samples_start, samples_end

    def initial_dataset(self, path="results/intial_distr.pdf"):
        plt.figure(figsize=(20, 20))
        plt.rc("text", usetex=True)
        plt.title("Initial Distribution", fontsize=45)
        data_grid = np.linspace(self.data.min(), self.data.max(), 1000)
        bandwidth = 0.1
        data_p_n = self._kde(self.data, data_grid, bandwidth=bandwidth)
        plt.fill_between(x=data_grid, y1=data_p_n, y2=0, alpha=0.7)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        plt.grid()
        savefig(path)
        plt.show()

    def compare_datasets(self, samples, title, ax):
        data_grid = np.linspace(self.data.min(), self.data.max(), 1000)
        sample_grid = np.linspace(samples.min(), samples.max(), 1000)
        bandwidth = 0.1
        data_p_n = self._kde(self.data, data_grid, bandwidth=bandwidth)
        sample_p_n = self._kde(samples, sample_grid, bandwidth=bandwidth)
        ax.fill_between(x=data_grid, y1=data_p_n, y2=0, alpha=0.7, label='real')
        ax.fill_between(x=sample_grid, y1=sample_p_n, y2=0, alpha=0.7, label='fake')
        ax.legend(prop={'size': 40})
        ax.grid()
        ax.set_title(title, fontsize=45)
        ax.tick_params(axis="x", labelsize=40)
        ax.tick_params(axis="y", labelsize=40)

    def epoch_vizual(self, train_logs, path):
        '''
        Training plots on each epoch
        '''
        fig = plt.figure(figsize=(20, 20))
        plt.rc("text", usetex=True)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.semilogy(train_logs["g_grad"], linewidth=7.0)
        ax1.set_title("Norm of Generator gradient", fontsize=45)
        ax1.set_xlabel('Training Iteration', fontsize=42)
        ax1.set_ylabel('Norm of gradient', fontsize=42)
        ax1.tick_params(axis="x", labelsize=40)
        ax1.tick_params(axis="y", labelsize=40)
        ax1.grid()

        ax2.semilogy(train_logs["c_grad"], linewidth=7.0)
        ax2.set_title("Norm of Discriminator gradient", fontsize=45)
        ax2.set_xlabel('Training Iteration', fontsize=42)
        ax2.set_ylabel('Norm of gradient', fontsize=42)
        ax2.tick_params(axis="x", labelsize=40)
        ax2.tick_params(axis="y", labelsize=40)
        ax2.grid()

        plt.savefig(path)
        plt.show()

    def eval(self, g, c):
        with torch.no_grad():
            data1 = torch.Tensor(self.dataset(n=250))
            data2 = g.sample(250)
            preds1 = c(data2).numpy()
            preds2 = c(data1).numpy()
            predictions_false = preds1 <= 0.5
            predictions_true = preds2 > 0.5
            correct = np.sum(predictions_false) + np.sum(predictions_true)
            accuracy = correct / (data1.shape[0] * 2)
            data1 = data1.numpy()
            data2 = data2.numpy()
            pvalue = stat.ks_2samp(data1.T[0], data2.T[0])[1]
            return {"pvalue": pvalue, "accuracy": accuracy}

    def final_plot(self, g, c, losses, samples_start, samples_end):
        clear_output(wait=True)
        fig = plt.figure(figsize=(30, 30))
        ax1 = fig.add_subplot(3, 2, 3)
        ax2 = fig.add_subplot(3, 2, 4)
        ax3 = fig.add_subplot(3, 2, 1)
        ax4 = fig.add_subplot(3, 2, 2)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)
        ax1.semilogy(losses["pvalue"], linewidth=7.0)
        ax1.set_title("p-value", fontsize=45)
        ax1.set_xlabel('Epoch', fontsize=42)
        ax1.set_ylabel('p-value', fontsize=42)
        ax1.tick_params(axis="x", labelsize=40)
        ax1.tick_params(axis="y", labelsize=40)
        ax1.grid()

        ax2.plot(losses["accuracy"], linewidth=7.0)
        ax2.set_title("Accuracy", fontsize=45)
        ax2.set_xlabel('Epoch', fontsize=42)
        ax2.set_ylabel('Accuracy', fontsize=42)
        ax2.tick_params(axis="x", labelsize=40)
        ax2.tick_params(axis="y", labelsize=40)
        ax2.grid()
        x = [600 * x for x in range(len(losses["g_grad"]))]
        ax5.semilogy(x, losses["g_grad"], linewidth=7.0)
        ax5.set_title("Norm of Generator gradient", fontsize=45)
        ax5.set_xlabel('Training Iteration', fontsize=42)
        ax5.set_ylabel('Norm of gradient', fontsize=42)
        ax5.tick_params(axis="x", labelsize=40)
        ax5.tick_params(axis="y", labelsize=40)
        ax5.grid()
        x = [600 * x for x in range(len(losses["c_grad"]))]
        ax6.semilogy(x, losses["c_grad"], linewidth=7.0)
        ax6.set_title("Norm of Discriminator gradient", fontsize=45)
        ax6.set_xlabel('Training Iteration', fontsize=42)
        ax6.set_ylabel('Norm of gradient', fontsize=42)
        ax6.tick_params(axis="x", labelsize=40)
        ax6.tick_params(axis="y", labelsize=40)
        ax6.grid()

        self.compare_datasets(samples_start, f'Epoch 1', ax3)
        self.compare_datasets(samples_end, f'Epoch 50', ax4)
        savefig(f'results/{self.config.exp_name}/{self.config.exp_name}.pdf')
        plt.show()


def plot_gan_training(losses, title, ax):
    n_itr = len(losses)
    xs = np.arange(n_itr)

    ax.plot(xs, losses, label='loss', linewidth=7.0)
    ax.set_ylim([-0.1, 0.1])
    ax.legend(prop={'size': 40})
    ax.grid()
    ax.set_title(title, fontsize=45)
    ax.set_xlabel('Training Iteration', fontsize=42)
    ax.set_ylabel('Loss', fontsize=42)
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)
