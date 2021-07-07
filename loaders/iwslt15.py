# import requests
# import torch 


# url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/'

# train_en = [line.split() for line in requests.get(url+"train.en").text.splitlines()]
# train_vi = [line.split() for line in requests.get(url+"train.vi").text.splitlines()]
# test_en = [line.split() for line in requests.get(url+"tst2013.en").text.splitlines()]
# test_vi = [line.split() for line in requests.get(url+"tst2013.vi").text.splitlines()]

# def make_vocab(train_data, min_freq):
#     vocab={}
#     for tokenlist in train_data:
#         for token in tokenlist:
#             if token not in vocab:
#                 vocab[token] = 0
#             vocab[token] += 1
#     vocablist = [('<unk>', 0), ('<pad>', 0), ('<cls>', 0), ('<eos>', 0)]
#     vocabidx = {}
#     for token, freq in vocab.items():
#         if freq >= min_freq:
#             idx = len(vocablist)
#             vocablist.append((token, freq))
#             vocabidx[token]= idx
#     vocabidx['<unk>']=0
#     vocabidx['<pad>']=1
#     vocabidx['<cls>']=2
#     vocabidx['<eos>']=3
#     return vocablist, vocabidx

# vocablist_en, vocabidx_en = make_vocab(train_en, 3)
# vocablist_vi, vocabidx_vi = make_vocab(train_vi, 3)

# def preprocess(data, vocabidx):
#   rr = []
#   for tokenlist in data:
#     tkl = ['<cls>']
#     for token in tokenlist:
#       tkl.append(token if token in vocabidx else '<unk>')
#     tkl.append('<eos>')
#     rr.append(tkl)
#   return rr

# train_en_prep = preprocess(train_en, vocabidx_en)
# train_vi_prep = preprocess(train_vi, vocabidx_vi)
# test_en_prep = preprocess(test_en, vocabidx_en)





