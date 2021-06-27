from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from libs.graphs.cnn import *

from utils.init_weight import init_model

class ImageClassifier(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        self.net = eval(cfg.net.name)(**cfg.net.kwargs)
        self.criterion = getattr(nn, cfg.criterion)()
        
        self.net = init_model(self.net, cfg.weight_init)

        self.example_input_array = torch.rand(1, *cfg.input_shape)

    def forward(self, x):
        _, logits = self.net(x)

        return logits

    def configure_optimizers(self):
        optimizer = getattr(optim, self.cfg.optim.name)(self.net.parameters(), **self.cfg.optim.kwargs)
        if 'scheduler' in self.cfg:
            scheduler = getattr(lr_scheduler, self.cfg.scheduler.name)(optimizer, **self.cfg.scheduler.kwargs)
        
            return {'optimizer': optimizer, 'scheduler': scheduler}
        return optimizer
    
    def training_step(self, train_batch, *args, **kwargs) -> STEP_OUTPUT:
        inputs, targets = train_batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        pred = outputs.max(1)[1]
        total = targets.numel()
        correct = pred.eq(targets.view_as(pred)).sum().item()
        return {'loss': loss, 'correct': correct, 'total': total}
    
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        correct, total = [], []
        for out in outputs:
            correct += [out['correct']]
            total += [out['total']]
        train_err = 100. * (1 - sum(correct) / sum(total))
        self.log('train_err', train_err, prog_bar=True, logger=True, on_epoch=True)

    def validation_step(self, batch, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        inputs, targets = batch
        outputs = self(inputs)
        pred = outputs.max(1)[1]

        total = targets.numel()
        correct = pred.eq(targets.view_as(pred)).sum().item()

        return {'correct': correct, 'total': total}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        correct, total = [], []
        for out in outputs:
            correct += [out['correct']]
            total += [out['total']]
        val_err = 100. * (1 - sum(correct) / sum(total))
        self.log('val_err', val_err, prog_bar=True, logger=True, on_epoch=True)
