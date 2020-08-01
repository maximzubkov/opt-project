from dataclasses import asdict
from math import ceil
from typing import Tuple, Dict, List

import numpy
import torch
from pytorch_lightning.core.lightning import LightningModule
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from configs import GANConfig
from models.modules import SN_Generator, SN_Discriminator, Generator, Discriminator


class GAN(LightningModule):
    def __init__(self, config: GANConfig, spec_norm=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

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
        return [g_optimizer, d_optimizer], []

    def generator_loss(self, generator, discriminator, x):
        """
        Wassersten loss for generator
        """
        fake_data = generator.sample(x.shape[0])
        return -discriminator(fake_data).mean()

    def discriminator_loss(self, generator, discriminator, x):
        """
        Wassersten loss for discriminator
        """
        fake_data = generator.sample(x.shape[0])
        return discriminator(fake_data).mean() - discriminator(x).mean()


    def _construnct_datatset(self, dataset_size: int):
        n_modes = len(self.config.dataset_params)
        gaussians = [
            numpy.random.normal(loc=loc, scale=scale, size=(dataset_size // n_modes,))
            for loc, scale in self.config.dataset_params
        ]
        data = 2 * MinMaxScaler().fit_transform(numpy.concatenate(gaussians).reshape(-1, 1) + 1) - 1
        return data

    # ===== TRAIN BLOCK =====

    def train_dataloader(self) -> DataLoader:
        dataset = self._construnct_datatset(self.config.train_dataset_size)
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        return train_loader


    def training_step(self, batch: PathContextBatch, batch_idx: int) -> Dict:
        # Dict str -> torch.Tensor [seq length; batch size * n_context]
        context = batch.context
        for k in context:
            context[k] = context[k].to(self.device)
        # [seq length; batch size]
        labels = batch.labels.to(self.device)

        # [seq length; batch size; vocab size]
        logits = self(context, batch.contexts_per_label, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)

        with torch.no_grad():
            subtoken_statistic = SubtokenStatistic.calculate_statistic(
                labels, logits.argmax(-1), [self.vocab.label_to_id[t] for t in [SOS, EOS, PAD, UNK]]
            )

        log = {"train/loss": loss}
        log.update(subtoken_statistic.calculate_metrics(group="train"))
        progress_bar = {"train/f1": log["train/f1"]}
        return {"loss": loss, "log": log, "progress_bar": progress_bar, "subtoken_statistic": subtoken_statistic}

    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        logs = {f"train/loss": torch.stack([out[loss_key] for out in outputs]).mean()}
        progress_bar = {k: v for k, v in logs.items() if k in [f"train/loss"]}
        return {"val_loss": logs[f"train/loss"], "log": logs, "progress_bar": progress_bar}


