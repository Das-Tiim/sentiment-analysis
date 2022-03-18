import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.legacy.vocab import Vocab
from collections import Counter
import random
import re
import numpy as np
from bow import BOW
from lstm import LSTM

# parameters
lr = 1e-3
batch_size = 100
max_document_length = 500
max_vocab_size = 5000
epochs = 5

# import dataset
train_iter, test_iter = IMDB(split=('train', 'test'))
train_list = list(train_iter)
test_list = list(test_iter)

# pre-processing
def remove_special_characters(text):
    special=r'[^a-zA-Z\s]'
    split=r'[a-z][A-Z]'
    text=re.sub(special,'',text)
    text=re.sub(split,' ',text)
    return text

tokenizer = get_tokenizer('basic_english')
counter = Counter()
for (sentiment, review) in train_list:
    review = remove_special_characters(review)
    counter.update(tokenizer(review))
for (sentiment, review) in test_list:
    review = remove_special_characters(review)
    counter.update(tokenizer(review))
vocab = Vocab(counter, max_size=max_vocab_size, min_freq=10, specials=('<unk>', '<PAD>'))
vocab_size = len(vocab)

text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
label_transform = lambda x: 1.0 if x == 'pos' else 0.0

def collate_batch(batch):
    label_list, text_list, len_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        len_list.append(processed_text.shape[0])
        text_list.append(processed_text)
    pad_text_list = torch.transpose(pad_sequence(text_list, padding_value=1.0), 0, 1)
    return torch.tensor(label_list), pad_text_list, torch.tensor(len_list)

def batch_sampler(list):
    indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(list)]
    random.shuffle(indices)
    pooled_indices = []
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
    pooled_indices = [x[0] for x in pooled_indices]
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

train_dataloader = DataLoader(train_list, batch_sampler=batch_sampler(train_list), collate_fn=collate_batch)
test_dataloader = DataLoader(test_list, batch_sampler=batch_sampler(test_list), collate_fn=collate_batch)



# training and evaluation
def accuracy(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item() / batch_size
  
def train(epochs, model, dataloader, optimizer, criterion):
    for epoch in range(0,epochs):
        batch_loss, batch_acc = [], []
        model.train()
        for i,batch in enumerate(dataloader,0):
            print(batch.size())
            optimizer.zero_grad()
            labels, text, lens  = batch
            if model == lstm_model:
                pred = model(text, lens).squeeze(1)
            else:
                pred = model(text).squeeze(1)
            loss = criterion(pred, labels)
            acc = accuracy(pred, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
            batch_acc.append(acc)
        epoch_loss = np.mean(batch_loss)
        epoch_acc = np.mean(batch_acc)
        print('Epoch:',epoch)
        print(f'\tTrain Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc * 100:.2f}%')

def evaluate(model, iterator, criterion):
    epoch_loss, epoch_acc = [], []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            labels, text, lens = batch
            if model == lstm_model:
                pred = model(text, lens).squeeze(1)
            else:
                pred = model(text).squeeze(1)
            loss = criterion(pred, labels)
            acc = accuracy(pred, labels)
            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
    return np.mean(epoch_loss), np.mean(epoch_acc)


def main(model, crit, lr):
    print(model)
    loss_func = crit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(epochs, model, train_dataloader, optimizer, loss_func)
    test_loss, test_acc = evaluate(model, test_dataloader, loss_func)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


bow_model = BOW(vocab_size, 500, 0.5)
bow_loss = nn.BCEWithLogitsLoss()
lstm_model = LSTM(vocab_size, 300, 128, 128, 1, 2, True, 0.5)
lstm_loss = nn.BCELoss()

main(bow_model,bow_loss, lr)
main(lstm_model, lstm_loss, lr)