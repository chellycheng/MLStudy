import string

import torch
import numpy as np
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from torchtext import data
from torchtext import datasets
import random
import os
import collections

# set up fields
RANDOM_SEED = 0
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.LabelField(dtype=torch.float)
"""
        Call get_dataset and makeImbalance in order to produce the imbalanced dataset
    """
"""
    Instruction:
    In main
        1. get_dataset()
        2. makeImbalance() with your path in directory
        3. get_vocab()  get the vocab file, the vocab should be around
            you could set keep tokens with > 5 occurrence by modifying min_occurance or choose the most common 50000 one 
        3. get_positivie_negative() generate the imbalanced data set according to the vocab
        3. get_positivie_negative_test() generate the balanced data set according to the vocab 
"""
def get_dataset():

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

def makeImbalance():
    fileDir = "/Users/hehuimincheng/Documents/GitHub/MLStudy/mini4/datanlp/imdb/aclImdb/train/neg"
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir,11250)
    for i in sample:
        os.remove(fileDir+"/"+i)
    print("File Removed!")

class load_imbalance_data_from_dictory():
    def load_doc(self,filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # turn a doc into clean tokens
    def clean_doc(self,doc,act):
        # split into tokens by white space
        tokens = doc.split()
        lemmatizer = WordNetLemmatizer()
        if act:
            # remove punctuation from each token
            table = str.maketrans('', '', string.punctuation)
            tokens = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            # filter out stop words
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if not w in stop_words]
            # filter out short tokens
            tokens = [word for word in tokens if len(word) > 1]
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens


    # load doc and add to vocab
    def add_doc_to_vocab(self,filename, vocab,act):
        # load doc
        doc = self.load_doc(filename)
        # clean doc
        tokens = self.clean_doc(doc, act)
        # update counts
        vocab.update(tokens)

    # load all docs in a directory
    def process_docs(self, directory, vocab,act):
        # walk through all files in the folder
        for filename in os.listdir(directory):
            # skip files that do not have the right extension
            if not filename.endswith(".txt"):
                continue
            # create the full path of the file to open
            path = directory + '/' + filename
            # add doc to vocab
            self.add_doc_to_vocab(path, vocab,act)

    # save list to file
    def save_list(self,lines, filename):
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()


    # load all docs in a directory
    def cat_docs(self,directory, vocab,act):
        lines = list()
        # walk through all files in the folder
        for filename in os.listdir(directory):
            # skip files that do not have the right extension
            if not filename.endswith(".txt"):
                continue
            # create the full path of the file to open
            path = directory + '/' + filename
            # load and clean the doc
            line = self.doc_to_line(path, vocab,act)
            # add to list
            lines.append(line)
        return lines


    # load doc, clean and return line of tokens
    def doc_to_line(self, filename, vocab,act):
        # load the doc
        doc = self.load_doc(filename)
        # clean doc
        tokens = self.clean_doc(doc,act)
        # filter by vocab
        tokens = [w for w in tokens if w in vocab]
        return ' '.join(tokens)

trainfileDir = "/Users/hehuimincheng/Documents/GitHub/MLStudy/mini4/datanlp/imdb/aclImdb/train"
testfileDir = "/Users/hehuimincheng/Documents/GitHub/MLStudy/mini4/datanlp/imdb/aclImdb/test"

def get_vocab():
    # define vocab
    vocab = collections.Counter()
    # add all docs to vocab
    ld= load_imbalance_data_from_dictory()
    ld.process_docs(trainfileDir+"/neg", vocab,True)
    ld.process_docs(trainfileDir+"/pos", vocab,True)
    # print the size of the vocab
    print(len(vocab))
    # print the top words in the vocab
    print(vocab.most_common(50))

    # keep tokens with > 5 occurrence
    # possible change here
    min_occurane = 5
    tokens = [k for k, c in vocab.items() if c >= min_occurane]

    tokens = [k for k, c in vocab.items()]
    print(len(tokens))
    # save tokens to a vocabulary file
    ld.save_list(tokens, 'vocab.txt')

def get_positivie_negative():
    vocab_filename = 'vocab.txt'
    ld = load_imbalance_data_from_dictory()
    vocab = ld.load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    # prepare negative reviews
    negative_lines = ld.cat_docs(trainfileDir + "/neg", vocab,True)
    #print(negative_lines)
    ld.save_list(negative_lines, 'negative.txt')
    # prepare positive reviews
    positive_lines = ld.cat_docs(trainfileDir + "/pos", vocab, True)
    ld.save_list(positive_lines, 'positive.txt')

def get_positivie_negative_test():
    vocab_filename = 'vocab.txt'
    ld = load_imbalance_data_from_dictory()
    vocab = ld.load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)
    # prepare negative reviews
    negative_lines = ld.cat_docs(testfileDir + "/neg", vocab,True)
    #print(negative_lines)
    ld.save_list(negative_lines, 'negative_test.txt')
    # prepare positive reviews
    positive_lines = ld.cat_docs(testfileDir + "/pos", vocab, True)
    ld.save_list(positive_lines, 'positive_test.txt')
def length_of_vocab():
    vocab_filename = 'vocab.txt'
    ld = load_imbalance_data_from_dictory()
    vocab = ld.load_doc(vocab_filename)
    vocab = vocab.split()
    print(len(vocab))

if __name__ == '__main__':
    length_of_vocab()