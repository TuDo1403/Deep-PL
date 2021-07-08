from torchtext.datasets import IWSLT2016
import requests
import torch
import re

from collections import Counter

# with open('data/vocab.vi.txt', 'r+', encoding='utf-8') as handle:
#     txt = handle.read()
#     clean = re.sub('\W+', ' ', txt.strip().lower())
#     test = Counter(clean.split(' '))
#     print(test)

# train_iter = IWSLT2016(split='train', language_pair=('en', 'de'))
# a = next(train_iter)

# dts = iter(list(torch.load('data/train_iter.pth.tar')))
# for sample in dts:
#     print(sample)
url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/'

train_en = [line.split() for line in requests.get(url+"train.en").text.splitlines()]
train_vi = [line.split() for line in requests.get(url+"train.vi").text.splitlines()]

def make_vocab(train_data, min_freq):
    vocab={}
    for tokenlist in train_data:
        for token in tokenlist:
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
    del_val = []
    for token, freq in vocab.items():
        if freq < min_freq:
            del_val += [token]

    for val in del_val:
        del vocab[val]
    vocab.update({
        '<unk>': 0, 
        '<pad>': 0, 
        '<cls>': 0, 
        '<eos>': 0
    })
    return vocab

vocab_en = make_vocab(train_en, 3)
vocab_vi = make_vocab(train_vi, 3)

print(len(vocab_en))
print(len(vocab_vi))

torch.save(vocab_en, 'data/vocab_en.pth.tar')
torch.save(vocab_vi, 'data/vocab_vi.pth.tar')
# # torch.save(train_en, 'train_en.pth,tar')
# # torch.save(train_vi, 'train_vi.pth.tar')

print('Debug')