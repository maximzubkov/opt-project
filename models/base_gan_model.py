from dataclasses import asdict
from typing import Tuple, Dict, List

import numpy
import scipy.stats as stat
import torch
from pytorch_lightning.core.lightning import LightningModule
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from configs import GANConfig
from models.modules import SN_Generator, SN_Discriminator, Generator, Discriminator
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class GAN(LightningModule):
    def __init__(self, config: GANConfig, num_workers: int, spec_norm=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_workers = num_workers

        generator_config = self.config.generator_config
        discriminator_config = self.config.discriminator_config
        if spec_norm:
            self.generator = SN_Generator(**asdict(generator_config))
            self.discriminator = SN_Discriminator(**asdict(discriminator_config))
        else:
            self.generator = Generator(**asdict(generator_config))
            self.discriminator = Discriminator(**asdict(discriminator_config))

    def forward(self, z) -> torch.Tensor:
        return self.generator(z)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        g_optimizer = Adam(self.generator.parameters(), lr=self.config.g_lr, betas=self.config.g_betas)
        d_optimizer = Adam(self.discriminator.parameters(), lr=self.config.d_lr, betas=self.config.d_betas)
        return [g_optimizer] + self.config.n_cr * [d_optimizer], []

    def generator_loss(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        Wassersten loss for generator
        """
        g_input = torch.randn(real_data.shape[0], self.config.generator_config.input_size).to(self.device)
        fake_data = self.generator(g_input)
        return -self.discriminator(fake_data).mean()

    def discriminator_loss(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        Wassersten loss for discriminator
        """
        real_data = real_data.to(self.device)
        input_noise = torch.randn(real_data.shape[0], self.config.generator_config.input_size, device=self.device)
        fake_data = self.generator(input_noise).detach()
        gradient_penalty = self._gradient_penalty(real_data, fake_data)
        return self.discriminator(fake_data).mean() - self.discriminator(real_data).mean() + gradient_penalty

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch_size = real.shape[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1).expand_as(real).to(self.device)

        interpolated = alpha * real.data + (1 - alpha) * fake.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.config.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _construnct_datatset(self, dataset_size: int):
        n_modes = len(self.config.dataset_params)
        gaussians = [
            numpy.random.normal(loc=loc, scale=scale, size=(dataset_size // n_modes,))
            for loc, scale in self.config.dataset_params
        ]
        data = numpy.concatenate(gaussians).reshape(-1, 1)
        return data.astype(numpy.float32)

    # ===== TRAIN BLOCK =====

    def train_dataloader(self) -> DataLoader:
        dataset = self._construnct_datatset(self.config.train_dataset_size)
        train_loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        return train_loader

    def training_step(self, real_data: torch.Tensor, batch_idx: int, optimizer_idx: int) -> Dict:
        if optimizer_idx == 0:
            g_loss = self.generator_loss(real_data)
            progress_bar = {"g_loss": g_loss}
            output = {"loss": g_loss, "progress_bar": progress_bar, "log": {"g_loss": g_loss}}
        else:
            d_loss = self.discriminator_loss(real_data)
            progress_bar = {"d_loss": d_loss}
            output = {"loss": d_loss, "progress_bar": progress_bar, "log": {"d_loss": d_loss}}
        output["log"].update(self.compute_metrcis(real_data))
        return output

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        logs = {
            "d_loss": torch.Tensor([out["log"]["d_loss"] for out in outputs if "d_loss" in out["log"]]).mean(),
            "g_loss": torch.Tensor([out["log"]["g_loss"] for out in outputs if "g_loss" in out["log"]]).mean(),
            "accuracy": torch.Tensor([out["log"]["acc"] for out in outputs if "acc" in out["log"]]).mean(),
            "pvalue": torch.Tensor([out["log"]["pvalue"] for out in outputs if "pvalue" in out["log"]]).mean(),
        }
        progress_bar = {k: v for k, v in logs.items() if k in ["d_loss", "g_loss", "accuracy", "pvalue"]}
        return {"d_loss": logs["d_loss"], "g_loss": logs["g_loss"], "log": logs, "progress_bar": progress_bar}

    def compute_metrcis(self, real: torch.Tensor) -> Dict:
        with torch.no_grad():
            real = real.to(self.device)
            input_noise = torch.randn(real.shape[0], self.config.generator_config.input_size, device=self.device)
            fake = self.generator(input_noise)
            preds_for_fake = self.discriminator(fake).numpy()
            preds_for_real = self.discriminator(real).numpy()
            correct = numpy.sum(preds_for_fake <= 0.5) + numpy.sum(preds_for_real > 0.5)
            accuracy = correct / (real.shape[0] + fake.shape[0])
            pvalue = stat.ks_2samp(real.numpy().T[0], fake.numpy().T[0])[1]
            return {"pvalue": pvalue, "acc": accuracy}
