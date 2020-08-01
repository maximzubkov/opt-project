from src import generator_w_loss, critic_w_loss
from .gan_config import GANConfig, MLPConfig


def get_gan_default_config(n_cr: int, exp_name: str, spec_norm: bool) -> GANConfig:
    c_config = MLPConfig(input_size=1, n_hidden=3, hidden_size=25, output_size=1, spec_norm=spec_norm)
    g_config = MLPConfig(input_size=3, n_hidden=3, hidden_size=25, output_size=1, spec_norm=spec_norm)
    return GANConfig(
        batch_size=64,
        n_epochs=50,
        n_cr=n_cr,

        c_lr=9e-5,
        c_betas=(0, 0.9),

        c_config=c_config,
        g_config=g_config,

        c_loss=critic_w_loss,
        g_loss=generator_w_loss,

        g_lr=9e-5,
        g_betas=(0, 0.9),

        exp_name=exp_name,
        dataset_params=[(2, 0.4), (0, 0.55), (5, 0.25)],
    )
