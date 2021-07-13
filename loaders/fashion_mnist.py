import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torchvision.datasets as datasets
from utils.neural_net import Cutout

import torch

class FashionMNIST:
    FASHION_MNIST_MEAN = (.1307,)
    FASHION_MNIST_STD = (.3081)
    def __init__(self, 
                 data_folder, 
                 num_workers, 
                 batch_size, 
                 pin_memory,
                 cutout=False,
                 cutout_length=None,
                 drop_last=False,
                 **kwargs):
        
        train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.FASHION_MNIST_MEAN, self.FASHION_MNIST_STD),
                                              torch.flatten])
        if cutout:
            print('Using Cut out: {}'.format(cutout_length))
            train_transform.transforms.append(Cutout(cutout_length))
        valid_transform = transforms.Compose([transforms.ToTensor(), 
                                              transforms.Normalize(self.FASHION_MNIST_MEAN, self.FASHION_MNIST_STD),
                                              torch.flatten])

        train_data = datasets.FashionMNIST(root=data_folder,
                                      train=True,
                                      download=True,
                                      transform=train_transform)

        test_data = datasets.FashionMNIST(root=data_folder,
                                     train=False,
                                     download=False,
                                     transform=valid_transform)
        
        self.train_loader = DataLoader(dataset=train_data,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       shuffle=True,
                                       drop_last=drop_last)
                                       
        self.test_loader = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      pin_memory=pin_memory,
                                      num_workers=num_workers,
                                      shuffle=False,
                                      drop_last=drop_last)