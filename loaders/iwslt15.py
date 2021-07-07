import requests
import torch 
BATCHSIZE = 128
def make_vocab(train_data, min_freq):
    vocab={}
    for tokenlist in train_data:
        for token in tokenlist:
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
    vocablist = [('<unk>', 0), ('<pad>', 0), ('<cls>', 0), ('<eos>', 0)]
    vocabidx = {}
    for token, freq in vocab.items():
        if freq >= min_freq:
            idx = len(vocablist)
            vocablist.append((token, freq))
            vocabidx[token]= idx
    vocabidx['<unk>']=0
    vocabidx['<pad>']=1
    vocabidx['<cls>']=2
    vocabidx['<eos>']=3
    return vocablist, vocabidx

def preprocess(data, vocabidx):
  rr = []
  for tokenlist in data:
    tkl = ['<cls>']
    for token in tokenlist:
      tkl.append(token if token in vocabidx else '<unk>')
    tkl.append('<eos>')
    rr.append(tkl)
  return rr

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

def padding_batch(b):
    maxlen = max([len(x) for x in b])
    for tokenlists in b:
        for i in range(maxlen - len(tokenlists)):
            tokenlists.append('<pad>')

    return b

def padding(bb):
    for ben, bvi in bb:
        ben = padding_batch(ben)
        bvi = padding_batch(bvi)
    return bb

if __name__ == "__main__":

  url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/'

  train_en = [line.split() for line in requests.get(url+"train.en").text.splitlines()]
  train_vi = [line.split() for line in requests.get(url+"train.vi").text.splitlines()]
  test_en = [line.split() for line in requests.get(url+"tst2013.en").text.splitlines()]
  test_vi = [line.split() for line in requests.get(url+"tst2013.vi").text.splitlines()]

  vocablist_en, vocabidx_en = make_vocab(train_en, 3)
  vocablist_vi, vocabidx_vi = make_vocab(train_vi, 3)
  
  train_en_prep = preprocess(train_en, vocabidx_en)
  train_vi_prep = preprocess(train_vi, vocabidx_vi)
  test_en_prep = preprocess(test_en, vocabidx_en)

  train_data = list(zip(train_en_prep, train_vi_prep))
  train_data.sort(key = lambda x: (len(x[0]), len(x[1])))
  test_data = list(zip(test_en_prep, test_en, test_vi))

  train_data_bb = make_batch(train_data, BATCHSIZE)

  train_data_pd = padding(train_data_bb)
  train_data_encoding=[([[vocabidx_en[token] for token in tokenlist] for tokenlist in ben],
             [[vocabidx_vi[token] for token in tokenlist] for tokenlist in bvi]) for ben, bvi in train_data_bb]
  test_data_encoding=[([vocabidx_en[token] for token in enprep],en ,vi) for enprep, en, vi in test_data]