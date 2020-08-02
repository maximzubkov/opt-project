from dataclasses import dataclass
from typing import Tuple, Callable, List


@dataclass(frozen=True)
class MLPConfig:
    input_size: int
    n_hidden: int
    hidden_size: int
    output_size: int


@dataclass(frozen=True)
class GANConfig:
    batch_size: int
    n_epochs: int
    n_cr: int
    train_dataset_size: int
    val_dataset_size: int

    d_lr: float
    d_betas: Tuple[float, float]

    discriminator_config: MLPConfig
    generator_config: MLPConfig

    g_lr: float
    g_betas: Tuple[float, float]

    dataset_params: List[Tuple[float, float]]

    save_every_epoch: int = 1
    val_every_epoch: int = 1
    log_every_epoch: int = 10
