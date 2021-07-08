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
from utils.masking import create_mask, generate_square_subsequent_mask




class LangTranslator(pl.LightningModule):
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    def __init__(self, 
                 cfg, 
                 text_transform, 
                 vocab_transform, 
                 src_lang,
                 tgt_lang,
                 *args: Any, 
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.cfg = cfg

        self.net = eval(cfg.net.name)(**cfg.net.kwargs)
        self.criterion = getattr(nn, cfg.criterion)(ignore_index=self.PAD_IDX)
        
        self.net = init_model(self.net, cfg.weight_init)
        self.text_transform = text_transform
        self.vocab_transform = vocab_transform

        self.SRC_LANGUAGE = src_lang
        self.TGF_LANGUAGE = tgt_lang
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
        targets_input = targets[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
            create_mask(inputs, targets_input, 1)

        logits = self(
            src=inputs,
            trg=targets_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        # output_l = torch.argmax(logits.detach().cpu(), dim=0).tolist()

        tgt_out = targets[1:, :]
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        self.log('train_loss', loss.item(), prog_bar=True, logger=True)

        return {'loss': loss}
    
    # def training_epoch_end(self, outputs) -> None:
    #     pred, ref = [], []
    #     for out in outputs:
    #         sentences = [[self.corpus_en[word][0] for word in sentence] for sentence in out['pred']]
    #         pred += [*sentences]
    #         ref += [*out['ref']]
    #     # pred = [[self.corpus_en[word][0] for word in sentence] for sentence in pred]
    #     score = bleu_score(pred, ref)
    #     self.log('bleu_score_train', score, prog_bar=True, logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        # prep_en = prep_en.squeeze(0).T; prep_vi = prep_vi.squeeze(0)
        # prep_en = torch.row_stack([torch.cat(row) for row in prep_en]).T
        # prep_vi = torch.row_stack([torch.cat(row) for row in prep_vi])

        # targets_input = prep_vi[:, :-1].T
        targets_input = targets[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
            create_mask(inputs, targets_input, 1)

        logits = self(
            src=inputs,
            trg=targets_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        # logits = logits.permute(1, 0, 2)
        output_l = torch.argmax(logits.detach().cpu(), dim=0).tolist()
        return {'pred': output_l, 'ref': self.vi[batch_idx]}

    def validation_epoch_end(self, outputs) -> None:
        pred, ref = [], []
        for out in outputs:
            sentences = [[self.corpus_en[word][0] for word in sentence] for sentence in out['pred']]
            pred += [*sentences]
            ref += [out['ref']]
        # pred = [[self.corpus_en[word][0] for word in sentence] for sentence in pred]
        score = bleu_score(pred, ref)
        self.log('bleu_score_val', score, prog_bar=True, logger=True, on_epoch=True)

    # function to generate output sequence using greedy algorithm 
    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        src = src.cuda()
        src_mask = src_mask.cuda()

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).cuda()
        for i in range(max_len-1):
            memory = memory.cuda()
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).cuda()
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys

    


    # actual function to translate input sentence into target language
    def translate(self, src):
        # self.net.eval()
        # src = self.text_transform[self.SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            self.net,  
            src, 
            src_mask, 
            max_len=num_tokens + 5, 
            start_symbol=self.BOS_IDX
        ).flatten()

        res = self.vocab_transform[self.TGF_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))
        for i in range(len(res)):
            if res[i] == '<eos>' or res[i] == '<bos>':
                res[i] = ''
        return res
        # return \
        #     " ".join(self.vocab_transform[self.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
