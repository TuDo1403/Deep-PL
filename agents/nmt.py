from types import coroutine
from typing import Any
import pytorch_lightning as pl

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchtext.data.metrics import bleu_score

from libs.graphs.sequence_model import *
from libs.graphs.attention_model import *

from utils.init_weight import init_model
from utils.masking import create_mask

class LangTranslator(pl.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        self.net = eval(cfg.net.name)(**cfg.net.kwargs)
        self.criterion = getattr(nn, cfg.criterion)()
        
        self.net = init_model(self.net, cfg.weight_init)
        self.corpus_en = torch.load('data/corpus_en.pth.tar')
        self.corpus_vi = torch.load('data/corpus_vi.pth.tar')
        # self.example_input_array = ({
        #     'src': torch.rand(93, 32),
        #     'trg': torch.rand(138, 32),
        #     'src_mask': torch.rand(93, 93).bool(),
        #     'tgt_mask': torch.rand(138, 138),
        #     'src_padding_mask': torch.rand(32, 93).bool(),
        #     'tgt_padding_mask': torch.rand(32, 138).bool(),
        #     'memory_key_padding_mask': torch.rand(32, 93).bool(),
        # })

    def forward(self, x):
        output = self.net(**x)

        return output

    def configure_optimizers(self):
        optimizer = getattr(optim, self.cfg.optim.name)(self.net.parameters(), **self.cfg.optim.kwargs)
        if 'scheduler' in self.cfg:
            scheduler = getattr(lr_scheduler, self.cfg.scheduler.name)(optimizer, **self.cfg.scheduler.kwargs)
        
            return {'optimizer': optimizer, 'scheduler': scheduler}
        return optimizer
    
    def training_step(self, train_batch, *args, **kwargs):
        inputs, targets = train_batch

        # inputs = torch.row_stack([torch.cat(row) for row in inputs]).T
        # targets = torch.row_stack([torch.cat(row) for row in targets])
        inputs = inputs.squeeze(0).T
        targets = targets.squeeze(0)

        targets_input = targets[:, :-1].T

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
            create_mask(inputs, targets_input, 1)

        logits = self({
            'src': inputs,
            'trg': targets_input,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_padding_mask': src_padding_mask,
            'tgt_padding_mask': tgt_padding_mask,
            'memory_key_padding_mask': src_padding_mask,
        })
        output_l = torch.argmax(logits, dim=0).tolist()

        tgt_out = targets[:, 1:].T
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {'loss': loss, 'pred': output_l, 'ref': targets}
    
    def training_epoch_end(self, outputs) -> None:
        pred, ref = [], []
        for out in outputs:
            sentences = [[self.corpus_en[word][0] for word in sentence] for sentence in out['pred']]
            pred += [*sentences]
            ref += [*out['ref']]
        # pred = [[self.corpus_en[word][0] for word in sentence] for sentence in pred]
        score = bleu_score(pred, ref)
        self.log('bleu_score_train', score, prog_bar=True, logger=True, on_epoch=True)

    def validation_step(self, batch, *args, **kwargs):
        prep_en, prep_vi, inputs, targets = batch
        prep_en = prep_en.squeeze(0).T; prep_vi = prep_vi.squeeze(0)
        # prep_en = torch.row_stack([torch.cat(row) for row in prep_en]).T
        # prep_vi = torch.row_stack([torch.cat(row) for row in prep_vi])

        targets_input = prep_vi[:, :-1].T

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
            create_mask(prep_en, targets_input, 1)

        logits = self({
            'src':prep_en,
            'trg':targets_input,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_padding_mask': src_padding_mask,
            'tgt_padding_mask': tgt_padding_mask,
            'memory_key_padding_mask': src_padding_mask,
        })
        output_l = torch.argmax(logits, dim=0).tolist()
        return {'pred': output_l, 'ref': targets}

    def validation_epoch_end(self, outputs) -> None:
        pred, ref = [], []
        for out in outputs:
            sentences = [[self.corpus_en[word][0] for word in sentence] for sentence in out['pred']]
            pred += [*sentences]
            ref += [*out['ref']]
        # pred = [[self.corpus_en[word][0] for word in sentence] for sentence in pred]
        score = bleu_score(pred, ref)
        self.log('bleu_score_val', score, prog_bar=True, logger=True, on_epoch=True)
