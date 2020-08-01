from src import generator_w_loss, critic_w_loss
from .gan_config import GANConfig, MLP_params


def get_gan_default_config(n_cr: int, exp_name: str, sn: bool) -> GANConfig:
    return GANConfig(
        batch_size=64,
        n_epochs=50,
        n_cr=n_cr,
        c_lr=9e-5,
        c_betas=(0, 0.9),
        c_params=(1, 3, 25, 1),
        c_loss=critic_w_loss,
        g_lr=9e-5,
        g_betas=(0, 0.9),
        g_params=(3, 3, 25, 1),
        g_loss=generator_w_loss,
        exp_name=exp_name,
        sn=sn,
        dataset_params=[(2, 0.4), (0, 0.55), (5, 0.25)],
    )
