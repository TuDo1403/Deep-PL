import random

import glob

import os

import torch
from torch.utils.data import Dataset

from PIL import Image
from torch.utils.data.dataset import T_co

class UnpairedImageDataset(Dataset):
    def __init__(self,
                 root,
                 mode,
                 transform=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.root = root; self.mode = mode
        self.transform = transform
        self.files_A = os.listdir(os.path.join(root, mode + 'A'))
        self.files_B = os.listdir(os.path.join(root, mode + 'B'))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A.copy()
        self.new_perm()
        assert len(self.files_A) > 0

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index) -> T_co:
        item_A = \
            self.transform(
                Image.open(
                    os.path.join(
                        self.root, 
                        self.mode + 'A',
                        self.files_A[index % len(self.files_A)]
                    )
                )
            )
        item_B = \
            self.transform(
                Image.open(
                    os.path.join(
                        self.root,
                        self.mode + 'B',
                        self.files_B[index % len(self.files_B)]
                    )
                )
            )
        if item_A.shape[0] != 3:
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3:
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()

        return item_A, item_B

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))
    
