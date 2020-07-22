from dataclasses import dataclass
from typing import Tuple, Callable, List

MLP_params = Tuple[int, int, int, int]


@dataclass(frozen=True)
class GANConfig:
    batch_size: int
    n_epochs: int
    n_cr: int
    c_lr: float
    c_betas: Tuple[float, float]
    c_params: MLP_params
    c_loss: Callable
    g_lr: float
    g_betas: Tuple[float, float]
    g_params: MLP_params
    g_loss: Callable
    exp_name: str
    sn: bool
    dataset_params: List[Tuple[float, float]]
