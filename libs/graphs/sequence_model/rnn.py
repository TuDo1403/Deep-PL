import torch
import torch.nn.functional as F
from torch import nn

# class RNN(nn.Module):
#   def __init__(self, vocab_size):
#     super(RNN, self).__init__()
#     self.emb = nn.Embedding(vocab_size, 300)
#     self.l1 = nn.Linear(300, 300)
#     self.l2 = nn.Linear(300, 2)

#   def forward(self, x):
#     e = self.emb(x)
#     h = torch.zeros(e[0].size(), dtype = torch.float32).cuda()
#     for i in range(x.size()[0]):
#       h = F.relu(e[i] + self.l1(h))
#     return self.l2(h)

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):

        #text = [sent len, batch size]
        
        embedded = self.embedding(text)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        
        output = self.fc(hidden.squeeze(0))
        # print('Output shape: {}'.format(output))
        return output.squeeze(0)