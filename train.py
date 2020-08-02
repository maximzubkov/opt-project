from argparse import ArgumentParser
from multiprocessing import cpu_count
from os.path import join

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger

from configs import get_gan_default_config, get_gan_test_config
from models import GAN

SEED = 7


def train(model_name: str, n_cr: int, num_workers: int = 0, is_test: bool = False, resume_from_checkpoint: str = None):
    seed_everything(SEED)

    if model_name == "improved_gan":
        config_function = get_gan_test_config if is_test else get_gan_default_config
        config = config_function(n_cr)
        model = GAN(config, num_workers, improved=True)
    elif model_name == "default_gan":
        config_function = get_gan_test_config if is_test else get_gan_default_config
        config = config_function(n_cr)
        model = GAN(config, num_workers, improved=False)
    else:
        raise ValueError(f"Model {model_name} is not supported")
    # define logger
    wandb_logger = WandbLogger(project="GAN", log_model=True, offline=is_test)
    wandb_logger.watch(model, log="all")
    # define model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(wandb.run.dir, "{epoch:02d}-{val_loss:.4f}"), period=config.save_every_epoch, save_top_k=3,
    )
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateLogger()
    trainer = Trainer(
        max_epochs=config.n_epochs,
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        row_log_interval=config.log_every_epoch,
        logger=wandb_logger,
        checkpoint_callback=model_checkpoint_callback,
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpu,
        callbacks=[lr_logger],
        reload_dataloaders_every_epoch=True,
    )

    trainer.fit(model)

    trainer.test()


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("model", choices=["sn_gan", "default_gan"])
    arg_parser.add_argument("--n_cr", type=int, default=2)
    arg_parser.add_argument("--n_workers", type=int, default=cpu_count())
    arg_parser.add_argument("--test", action="store_true")
    arg_parser.add_argument("--resume", type=str, default=None)
    args = arg_parser.parse_args()

    train(args.model, args.n_cr, args.n_workers, args.test, args.resume)
