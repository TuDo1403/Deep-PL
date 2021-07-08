import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from utils.load_cfg import load_cfg
from utils.prepare_seed import prepare_seed

import click

from agents import *

from loaders import *

@click.command()
@click.option('--config', '-cfg', required=True)
def cli(config):
    cfg = load_cfg(config)
    prepare_seed(cfg.exp_cfg.seed)
    agent = eval(cfg.agent)(cfg.agent_cfg)
    loaders = eval(cfg.data_loader.name)(**cfg.data_loader.kwargs)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        **cfg.model_checkpoint
    )
    logger = TensorBoardLogger(
        name=cfg.exp_name,
        **cfg.logger
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.out_dir,
        logger=logger,
        **cfg.trainer
    )

    trainer.fit(
        model=agent,
        train_dataloader=loaders.train_loader,
        val_dataloaders=loaders.test_loader
    )

if __name__ == '__main__':
    # cli(['-cfg', 'configs/iwslt15_transformer.yaml'])
    cli()
