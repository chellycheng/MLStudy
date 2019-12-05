import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from IMDB_loader import IMDB
import numpy as np
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_length):
        # [sentence len, batch size] => [sentence len, batch size, embedding size]
        embedded = self.embedding(text)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length)

        # [sentence len, batch size, embedding size] =>
        #  output: [sentence len, batch size, hidden size]
        #  hidden: [1, batch size, hidden size]
        # change packed to packed.data
        packed_output, hidden = self.rnn(packed.data)

        return self.fc(hidden.squeeze(0)).view(-1)

RANDOM_SEED=0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1
#include UNK
INPUT_DIM = 98950
torch.manual_seed(RANDOM_SEED)
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())

#optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

"""Tarning Start"""

def compute_binary_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            text, text_lengths = batch_data.text
            logits = model(text, text_lengths)
            predicted_labels = (torch.sigmoid(logits) > 0.5).long()
            num_examples += batch_data.label.size(0)
            correct_pred += (predicted_labels == batch_data.label.long()).sum()
        return float(correct_pred)/num_examples * 100

def train():
    train_set = IMDB(True)
    batch_size = 128
    test_set = IMDB(False)

    num_train = len(train_set)
    indices = list(range(num_train))
    train_portion = 0.8
    split = int(np.floor(train_portion * num_train))

    train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                  num_workers=4)
    valid_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
                                  num_workers=4)

    test_loader =  DataLoader(test_set,batch_size=batch_size, num_workers=4)

    start_time = time.time()
    NUM_EPOCHS = 100
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            text, text_lengths = batch_data.text

            ### FORWARD AND BACK PROP
            logits = model(text, text_lengths)
            cost = F.binary_cross_entropy_with_logits(logits, batch_data.label)
            optimizer.zero_grad()

            cost.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                      f'Cost: {cost:.4f}')

        with torch.set_grad_enabled(False):
            print(f'training accuracy: '
                  f'{compute_binary_accuracy(model, train_loader, DEVICE):.2f}%'
                  f'\nvalid accuracy: '
                  f'{compute_binary_accuracy(model, valid_loader, DEVICE):.2f}%')

        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
    print(f'Test accuracy: {compute_binary_accuracy(model, test_loader, DEVICE):.2f}%')
