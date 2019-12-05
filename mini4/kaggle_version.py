import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchtext import data
from torchtext import datasets
from keras.preprocessing.text import Tokenizer

class load_imbalance_data_from_dictory():
    def load_doc(self,filename1,filename2):
        # open the file as read only
        file1 = open(filename1, 'r')
        # read all text
        texts1 = file1.readlines()
        # close the file
        file1.close()

        file2 = open(filename2, 'r')
        # read all text
        texts2 = file2.readlines()
        # close the file
        file2.close()

        length1 = len(texts1)
        length2 = len(texts2)
        labels = [1]*length1 + [0] *length2
        texts = texts1+ texts2
        print(len(texts))
        print("labels",len(labels))
        return texts,labels

    # load all docs in a directory

class IMDB():

    def __init__(self,  train):
        if train :
            # get the pos

            ld = load_imbalance_data_from_dictory()
            fileName1 = "/kaggle/input/negative.txt"
            fileName2 = "/kaggle/input/positive.txt"
            texts, label = ld.load_doc(fileName1, fileName2)

            self.train_data= texts
            self.labels =label
        else:

            ld = load_imbalance_data_from_dictory()
            fileName1 = "/kaggle/input/negative_test.txt"
            fileName2 = "/kaggle/input/positive_test.txt"
            texts, label = ld.load_doc(fileName1, fileName2)

            self.train_data = texts
            self.labels = label

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        label = -1;

        if self.labels is not None:
            label = self.labels[index]

        return self.train_data[index], label

trainfileDir = "/Users/hehuimincheng/Documents/GitHub/MLStudy/mini4"



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


"""Tarning Start"""

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
#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

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

def prepare_data_train(train_docs, mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    # encode training data set
    return Xtrain

def prepare_data_test(train_docs,test_docs,mode):
    # create the tokenizer
    tokenizer = Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs)
    # encode training data set
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    # encode training data set
    return Xtest
class IMDB():

    def __init__(self,  train):

        #modes = ['binary', 'count', 'tfidf', 'freq']

        ld = load_imbalance_data_from_dictory()
        fileName1 = "/kaggle/input/negative.txt"
        fileName2 = "/kaggle/input/positive.txt"
        texts1, label = ld.load_doc(fileName1, fileName2)

        if train :
            # get the pos

            Xtrain = prepare_data_train(texts1, 'tfidf')
            self.train_data= Xtrain
            self.labels =label
        else:

            ld = load_imbalance_data_from_dictory()
            fileName1 = "/kaggle/input/negative_test.txt"
            fileName2 = "/kaggle/input/positive_test.txt"
            texts, label = ld.load_doc(fileName1, fileName2)
            Xtest = prepare_data_test(texts1,texts, 'tfidf')
            self.train_data = Xtest
            self.labels = label

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):


        data = self.train_data[index]
        label = self.labels[index]

        return data, label


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

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

    start_time = time.time()
    NUM_EPOCHS = 100
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (text, label) in enumerate(train_loader):

            text_lengths = len(text)

            ### FORWARD AND BACK PROP
            # print(text_lengths)
            # print(text)
            # print(label)

            logits = model(text, text_lengths)
            cost = F.binary_cross_entropy_with_logits(logits, label)
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