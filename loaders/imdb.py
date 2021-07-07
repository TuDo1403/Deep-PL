import torch
from torch.utils.data import DataLoader, TensorDataset

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences



class IMDB:
    def __init__(self,
                 root='./data',
                 batch_size=64,
                 num_workers=4,
                 pin_memory=True,
                 input_dim=2000,
                 max_len=200,
                 **kwargs) -> None:
        (x_train, y_train), (x_val, y_val) = imdb.load_data(path=root, num_words=input_dim)
        x_train = pad_sequences(x_train, maxlen=max_len)
        x_val = pad_sequences(x_val, maxlen=max_len)

        train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        valid_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        self.train_loader = DataLoader(
            train_data, 
            shuffle=True, 
            batch_size=batch_size, 
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.test_loader = DataLoader(
            valid_data, 
            shuffle=False, 
            batch_size=batch_size, 
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    @staticmethod
    def recover_text(sample, index_to_word):
        return ' '.join([index_to_word[i] for i in sample])

    @staticmethod
    def make_vocab(train_data, min_freq):
        vocab = {}
        for label, tokenlist in train_data:
            for token in tokenlist:
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
        vocab_l = [('<unk>', 0), ('<pad>', 0), ('<cls>', 0), ('<eos>', 0)]
        vocab_idx = {}
        for token, freq in vocab.items():
            if freq >= min_freq:
                idx = len(vocab_l)
                vocab_l.append((token, freq))
                vocab_idx[token] = idx
        vocab_idx['<unk>'] = 0
        vocab_idx['<pad>'] = 1
        vocab_idx['<cls>'] = 2
        vocab_idx['<eos>'] = 3
        return vocab_l, vocab_idx

    @staticmethod
    def preprocess(data, vocab_idx):
        rr = []
        for label, tokenlist in data:
            tkl = ['<cls>']
            for token in tokenlist:
                tkl.append(token if token in vocab_idx else '<unk>')
                tkl.append('<eos>')
                rr.append((label, tkl))
        return rr

    @staticmethod
    def make_batch(data, batch_size):
        bb = []
        b_label = []
        b_token_l = []
        for label, tokenlist in data:
            b_label.append(label)
            b_token_l.append(tokenlist)
            if len(b_label) >= batch_size:
                bb.append((b_token_l, b_label))
                b_label = []
                b_token_l = []
        if len(b_label) > 0:
            bb.append((b_label, b_label))
        return bb

    @staticmethod
    def padding(bb):
        for tokenlists, labels in bb:
            maxlen = max([len(x) for x in tokenlists])
            for tkl in tokenlists:
                for i in range(maxlen - len(tkl)):
                    tkl.append('<pad>')
        return bb


    @staticmethod
    def word2id(bb, vocab_idx):
        rr = []
        for tokenlists, labels in bb:
            id_labels = [1 if label == 'pos' else 0 for label in labels]
            id_tokenlists = []
            for tokenlist in tokenlists:
                id_tokenlists.append([vocab_idx[token] for token in tokenlist])
            rr.append((id_tokenlists, id_labels))
        return rr