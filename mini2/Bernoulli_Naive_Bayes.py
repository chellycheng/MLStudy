# import some library
# -*- coding: utf-8 -*
import numpy as np
import scipy as scp
import random
import string
import pandas as pd
#model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
#help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#help_feature_process
from sklearn.feature_extraction.text import CountVectorizer
#help_analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#analysis
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
import sklearn.naive_bayes
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statistics import mean

class Initial:
    def initiation():
        data = pd.read_csv('reddit_train.csv',names=["id","comments","subreddits"],errors='ignore')
        #test_sentsp = data.shuffle()
        return data

class Reader:
    def __init__(self, path):
        # open the file
        self.file = open(path, errors='ignore', encoding='utf-8')
        self.data = []
        # shuffle the documents at firsts

    def shuffle(self):
        data_p = [(random.random(), line) for line in self.file]
        data_p.sort()
        sample_sets = [str(w) for (a, w) in data_p]
        self.data = sample_sets
        return sample_sets


    def write(self,name):
        f = open(name, "w")
        for line in self.data:
            f.write(line)
        f.close()

    def reading(self):
        data_p = [ (line) for line in self.file]
        return data_p

    def close_file(self):
        self.file.close()

class classifier:
    def __init__(self, x_train, y_train, x_test, y_test, ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic(self, c):
        model = LogisticRegression(C=c, dual=True, solver='liblinear')
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        # accuracy = (preds == self.y_test).mean()
        # print("Losistic Regression : accurancy_mean is", accuracy)
        scores1 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of Logistic in Cross Validation", scores1.mean() * 100)
        print("Losistic Regression : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n", cm)
        print("Report", classification_report(self.y_test, preds))

    def NaiveB(self, alpha):
        model = BernoulliNB(alpha=alpha).fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores2 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of Naive Bayes", scores2.mean() * 100)
        print("Naive Bayes : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n", cm)
        print("Report", classification_report(self.y_test, preds))

    def svm(self, c):
        model = LinearSVC(C=c)
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)

        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of SVM in Cross Validation", scores3.mean() * 100)
        print("SVM Regression : accurancy_is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n", cm)
        print("Report", classification_report(self.y_test, preds))

    def dummy(self):
        clf = DummyClassifier(strategy='stratified', random_state=0)
        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        print("Random Baseline's accurancy", score)
    # 6 Useful analysis method
def main():
    data = pd.read_csv('reddit_train.csv',names=["id","comments","subreddits"],error_bad_lines=False,encoding='utf-8')

if __name__ == "__main__":
    main()