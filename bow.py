import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BOW(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units, dropout_prob, pad_index = 1):
        super(BOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units, padding_idx=pad_index)
        self.fc1 = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        self.bn = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(no_of_hidden_units, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, text):
        bow_embedding = []
        for i in range(len(text)):
            lookup_tensor = Variable(torch.LongTensor(text[i]))
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)

        x_ = self.fc1(bow_embedding)
        x_ = self.bn(x_)
        x_ = F.relu(x_)
        x_ = self.dropout(x_)
        x_ = self.fc2(x_)
        x = self.sig(x_)

        return x