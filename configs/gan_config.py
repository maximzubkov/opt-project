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

    d_optimizer: str
    d_lr: float
    d_betas: Tuple[float, float]

    discriminator_config: MLPConfig
    generator_config: MLPConfig

    d_loss: Callable
    g_loss: Callable

    g_optimizer: str
    g_lr: float
    g_betas: Tuple[float, float]

    exp_name: str
    dataset_params: List[Tuple[float, float]]
