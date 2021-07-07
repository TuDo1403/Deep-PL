from typing import Any
import pytorch_lightning as pl

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from libs.graphs.sequence_model import *
from libs.graphs.attention_model import *

from utils.init_weight import init_model

class RNNSentimentAnalyzer(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        self.net = eval(cfg.net.name)(**cfg.net.kwargs)
        self.criterion = getattr(nn, cfg.criterion)()
        
        self.net = init_model(self.net, cfg.weight_init)

        self.example_input_array = torch.rand(1, *cfg.input_shape).long()

    def forward(self, x):
        output = self.net(x)

        return output

    def configure_optimizers(self):
        optimizer = getattr(optim, self.cfg.optim.name)(self.net.parameters(), **self.cfg.optim.kwargs)
        if 'scheduler' in self.cfg:
            scheduler = getattr(lr_scheduler, self.cfg.scheduler.name)(optimizer, **self.cfg.scheduler.kwargs)
        
            return {'optimizer': optimizer, 'scheduler': scheduler}
        return optimizer
    
    def training_step(self, train_batch, *args, **kwargs):
        inputs, targets = train_batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets.view_as(outputs).float())

        rounded_pred = torch.round(torch.sigmoid(outputs))
        correct = rounded_pred.eq(targets.view_as(rounded_pred)).sum().item()
        total = targets.numel()
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss, 'correct': correct, 'total': total}
    
    def training_epoch_end(self, outputs) -> None:
        correct, total = [], []
        for out in outputs:
            correct += [out['correct']]
            total += [out['total']]
        train_err = 100. * (1 - sum(correct) / sum(total))
        self.log('train_err', train_err, prog_bar=True, logger=True, on_epoch=True)
        self.log('train_acc', 100 - train_err, prog_bar=True, logger=True, on_epoch=True)

    def validation_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        outputs = self(inputs)
        rounded_pred = torch.round(torch.sigmoid(outputs))
        correct = rounded_pred.eq(targets.view_as(rounded_pred)).sum().item()
        total = targets.numel()
        return {'correct': correct, 'total': total}

    def validation_epoch_end(self, outputs) -> None:
        correct, total = [], []
        for out in outputs:
            correct += [out['correct']]
            total += [out['total']]
        val_err = 100. * (1 - sum(correct) / sum(total))
        self.log('val_err', val_err, prog_bar=True, logger=True, on_epoch=True)
        self.log('val_acc', 100 - val_err, prog_bar=True, logger=True, on_epoch=True)
