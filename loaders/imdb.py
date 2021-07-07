import torch

from torchtext.legacy import datasets
from torchtext.legacy import data


class IMDB:
    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    LABEL = data.LabelField(dtype=torch.float)
    MAX_VOCAB_SIZE = 25_000

    def __init__(self,
                 root,
                 batch_size,
                 **kwargs) -> None:
        train_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL, root=root)
        self.TEXT.build_vocab(train_data, max_size = self.MAX_VOCAB_SIZE)
        self.LABEL.build_vocab(train_data)
        self.train_loader, self.test_loader = \
            data.BucketIterator.splits(
            (train_data, test_data), 
            batch_sizes=(batch_size, batch_size)
        )