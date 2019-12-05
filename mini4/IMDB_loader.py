from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import itertools
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
    def load_doc_vocab(self,filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text
    # load all docs in a directory

vocab_filename = 'vocab.txt'
ld = load_imbalance_data_from_dictory()
vocab = ld.load_doc_vocab(vocab_filename)
vocab = vocab.split()


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
        fileName1 = "negative.txt"
        fileName2 = "positive.txt"
        texts1, label = ld.load_doc(fileName1, fileName2)

        if train :
            # get the pos

            Xtrain = prepare_data_train(texts1, 'tfidf')
            self.train_data= Xtrain
            self.labels =label
        else:

            ld = load_imbalance_data_from_dictory()
            fileName1 = "negative_test.txt"
            fileName2 = "positive_test.txt"
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

trainfileDir = "/Users/hehuimincheng/Documents/GitHub/MLStudy/mini4"


if __name__ == '__main__':
        #print(label)