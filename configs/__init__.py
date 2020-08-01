from src import generator_w_loss, critic_w_loss
from .gan_config import GANConfig, MLPConfig


def get_gan_default_config(n_cr: int, exp_name: str) -> GANConfig:
    discriminator_config = MLPConfig(input_size=1, n_hidden=3, hidden_size=25, output_size=1)
    generator_config = MLPConfig(input_size=3, n_hidden=3, hidden_size=25, output_size=1)
    return GANConfig(
        batch_size=64,
        n_epochs=50,
        n_cr=n_cr,
        train_dataset_size=20000,
        val_dataset_size=5000,
        d_lr=9e-5,
        d_betas=(0, 0.9),
        discriminator_config=discriminator_config,
        generator_config=generator_config,
        d_loss=critic_w_loss,
        g_loss=generator_w_loss,
        g_lr=9e-5,
        g_betas=(0, 0.9),
        exp_name=exp_name,
        dataset_params=[(2, 0.4), (0, 0.55), (5, 0.25)],
    )
