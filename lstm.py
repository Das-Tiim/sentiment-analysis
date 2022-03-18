import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, hidden_dim , num_classes, lstm_layers,
                 bidirectional, dropout, pad_index = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_units,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.fc1 = nn.Linear(lstm_units * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units
        self.sig = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_units)),
                Variable(torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_units)))
        return h, c

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, _ = pad_packed_sequence(output, batch_first=True)
        x_ = output_unpacked[:, -1, :]
        x_ = self.relu(x_)
        x_ = self.fc1(x_)
        x_ = self.dropout(x_)
        x_ = self.fc2(x_)
        x = self.sig(x_)
        return x