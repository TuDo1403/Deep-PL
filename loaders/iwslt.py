from typing import List
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, vocab

from collections import OrderedDict




class IWSLT15:
    SRC_LANGUAGE = 'en'
    TGT_LANGUAGE = 'vi'
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 pin_memory,
                 **kwargs) -> None:
        self.token_transform = {
            self.SRC_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
            self.TGT_LANGUAGE: get_tokenizer('spacy', language='vi_core_news_lg')
        }
        self.vocab_transform = {}
        file_paths = ['data/vocab_en.pth.tar', 'data/vocab_vi.pth.tar']
        for i, ln in enumerate([self.SRC_LANGUAGE, self.TGT_LANGUAGE]):
            counter = torch.load(file_paths[i])
            sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
            self.vocab_transform[ln] = vocab(ordered_dict)
        # for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
        #     self.vocab_transform[ln].set_default_index(self.UNK_IDX)
        print(len(self.vocab_transform[self.SRC_LANGUAGE]))
        print(len(self.vocab_transform[self.TGT_LANGUAGE]))
        # assert len(self.vocab_transform[self.SRC_LANGUAGE]) == 24416
        # assert len(self.vocab_transform[self.TGT_LANGUAGE]) == 10662

        # src and tgt language text transforms to convert raw strings into tensors indices
        self.text_transform = {}
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.text_transform[ln] = self.sequential_transforms(
                self.token_transform[ln], #Tokenization
                self.vocab_transform[ln], #Numericalization
                self.tensor_transform
            )

        train_iter = list(torch.load('data/train_iter.pth.tar'))
        val_iter = list(torch.load('data/test_iter.pth.tar'))

        self.train_loader = DataLoader(
            train_iter,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers
        )

        self.test_loader = DataLoader(
            val_iter,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers
        )

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch

    @staticmethod
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func


    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]), 
                        torch.tensor(token_ids), 
                        torch.tensor([self.EOS_IDX])))