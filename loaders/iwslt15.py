from os import stat
import requests
from torch.autograd.grad_mode import F

import torch
from torch.utils.data import DataLoader
  

import numpy as np
from torch.utils.data.dataset import TensorDataset

class IWSLT15:
    def __init__(self, 
                 batch_size=128,
                 pin_memory=True,
                 num_workers=4,
                 min_freq=3,
                 **kwargs ) -> None:

        # url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/'

        # train_en = [line.split() for line in requests.get(url+"train.en").text.splitlines()]
        # train_vi = [line.split() for line in requests.get(url+"train.vi").text.splitlines()]
        # test_en = [line.split() for line in requests.get(url+"tst2013.en").text.splitlines()]
        # test_vi = [line.split() for line in requests.get(url+"tst2013.vi").text.splitlines()]

        # print('Making vocab')
        # vocablist_en, vocab_idx_en = self.make_vocab(train_en, min_freq)
        # vocablist_vi, vocab_idx_vi = self.make_vocab(train_vi, min_freq)

        # print('VI len: {}'.format(len(vocablist_vi)))
        # print('EN len: {}'.format(len(vocablist_en)))

        # # torch.save(vocablist_en, 'data/corpus_en.pth.tar')
        # # torch.save(vocablist_vi, 'data/corpus_vi.pth.tar')
        
        # print('Preprocessing')
        # train_en_prep = self.preprocess(train_en, vocab_idx_en)
        # train_vi_prep = self.preprocess(train_vi, vocab_idx_vi)
        # test_en_prep = self.preprocess(test_en, vocab_idx_en)
        # test_vi_prep = self.preprocess(test_vi, vocab_idx_vi)

        # train_data = list(zip(train_en_prep, train_vi_prep))
        # train_data.sort(key = lambda x: (len(x[0]), len(x[1])))
        # test_data = list(zip(test_en_prep, test_vi_prep, test_en, test_vi))

        # full_batch_train = len(train_data)
        # full_batch_test = len(test_data)


        # print('Batch making')
        # train_data_bb = self.make_batch(train_data, full_batch_train)
        # test_data_bb = self.make_batch_test(test_data, full_batch_test)

        # print('Padding')
        # train_data_pd = self.padding(train_data_bb)
        # test_data_pd = self.padding(test_data_bb)
        # train_data_encoding = [(
        #     torch.from_numpy(np.array([[vocab_idx_en[token] for token in tokenlist] for tokenlist in ben])),
        #     torch.from_numpy(np.array([[vocab_idx_vi[token] for token in tokenlist] for tokenlist in bvi]))) for ben, bvi in train_data_pd
        # ]
        # test_data_encoding = [(
        #     torch.from_numpy(np.array([[vocab_idx_en[token] for token in tokenlist] for tokenlist in en_prep])),
        #     torch.from_numpy(np.array([[vocab_idx_vi[token] for token in tokenlist] for tokenlist in vi_prep])), 
        #     en,
        #     vi) for en_prep, vi_prep, en, vi in test_data_pd
        # ]

        # torch.save(train_data_encoding, 'data/iwslt15_train_32.pth.tar')
        # torch.save(test_data_encoding, 'data/iwslt15_test_32.pth.tar')

        # train_data_encoding = torch.load('data/iwslt15_train_32.pth.tar')
        # test_data_encoding = torch.load('data/iwslt15_test_32.pth.tar')

        # train_data_encoding[0] = torch.from_numpy(np.array(train_data_encoding[0]))
        # train_data_encoding[1] = torch.from_numpy(np.array(train_data_encoding[1]))
        # test_data_encoding[0] = torch.from_numpy(np.array(test_data_encoding[0]))
        # test_data_encoding[1] = torch.from_numpy(np.array(test_data_encoding[1]))

        train_set = torch.load('data/train_set.pth.tar')
        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

        test_set = torch.load('data/test_set.pth.tar')
        self.test_loader = DataLoader(
            test_set,
            batch_size=1,
            drop_last=False,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers
        )

        print('Prepare Data Done!')
    
    @staticmethod
    def make_vocab(train_data, min_freq):
        vocab={}
        for tokenlist in train_data:
            for token in tokenlist:
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
        vocablist = [('<unk>', 0), ('<pad>', 0), ('<cls>', 0), ('<eos>', 0)]
        vocab_idx = {}
        for token, freq in vocab.items():
            if freq >= min_freq:
                idx = len(vocablist)
                vocablist.append((token, freq))
                vocab_idx[token]= idx
        vocab_idx['<unk>']=0
        vocab_idx['<pad>']=1
        vocab_idx['<cls>']=2
        vocab_idx['<eos>']=3
        return vocablist, vocab_idx

    @staticmethod
    def preprocess(data, vocab_idx):
        rr = []
        for tokenlist in data:
            tkl = ['<cls>']
            for token in tokenlist:
                tkl.append(token if token in vocab_idx else '<unk>')
                tkl.append('<eos>')
            rr.append(tkl)
        return rr

    @staticmethod
    def make_batch(data, batchsize):
        bb = []
        ben = []
        bvi = []
        for en, vi in data:
            ben.append(en)
            bvi.append(vi)
            if len(ben) >= batchsize:
                bb.append((ben, bvi))
                ben = []
                bvi = []
        if len(ben) > 0:
            bb.append((ben, bvi))
        return bb

    @staticmethod
    def make_batch_test(data, batchsize):
        bb = []

        ben = []
        bvi = []
        preps_en = []; preps_vi = []
        for prep_en, prep_vi, en, vi in data:
            ben.append(en)
            bvi.append(vi)
            preps_en += [prep_en]; preps_vi += [prep_vi]
            if len(ben) >= batchsize:
                bb.append((preps_en, preps_vi, ben, bvi))
                ben = []
                bvi = []
                preps_en = []; preps_vi = []
        if len(ben) > 0:
            bb.append((preps_en, preps_vi, ben, bvi))
        return bb
    
    @staticmethod
    def padding_batch(b):
        maxlen = max([len(x) for x in b])
        for tokenlists in b:
            for i in range(maxlen - len(tokenlists)):
                tokenlists.append('<pad>')

        return b

    def padding(self, bb):
        for attrs in bb:
            for i, att in enumerate(attrs):
                if i <= 1:
                    att = self.padding_batch(att)
        return bb

    # def padding(self, bb):
    #     for attrs in bb:
    #         for i, att in enumerate(attrs):
    #             if i <= 1:
    #                 att = self.padding_batch(att)
    #         # ben = self.padding_batch(ben)
    #         # bvi = self.padding_batch(bvi)
    #     return bb
