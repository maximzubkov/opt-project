from dataclasses import dataclass
from typing import Tuple, Callable, List


@dataclass(frozen=True)
class MLPConfig:
    input_size: int
    n_hidden: int
    hidden_size: int
    output_size: int
    spec_norm: bool


@dataclass(frozen=True)
class GANConfig:
    batch_size: int
    n_epochs: int
    n_cr: int

    c_lr: float
    c_betas: Tuple[float, float]

    c_config: MLPConfig
    g_config: MLPConfig

    c_loss: Callable
    g_loss: Callable

    g_lr: float
    g_betas: Tuple[float, float]

    exp_name: str
    dataset_params: List[Tuple[float, float]]
