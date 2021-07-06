from torch import nn

import torch.functional as F

import torch

class CIFAR_conv(nn.Module):
  def __init__(self):
    super(CIFAR_conv,self).__init__()
    self.l1 = nn.Conv2d(3,3,5)#(H,W,filter) #Co the thay doi kich thuoc in_channel, out_channel, bo loc
    self.l2 = nn.Linear(3*28*28, 300)#OH = (H + 2P -FH)+1 /S https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    self.l3 = nn.Linear(300,10) #
  def forward(self,x):  
    h = F.relu(self.l1(x))
    h  = torch.flatten(h, start_dim=1)#flatten
    h = F.relu(self.l2(h))
    y = self.l3(h)
    return y