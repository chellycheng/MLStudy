import sklearn
import numpy as np
import scipy.misc # to visualize only
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm, linear_model, naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import skimage
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.feature import daisy
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


def extract():
    y = pd.read_csv('train_max_y.csv',encoding='utf-8')
    y_id = y['Id']
    y_lable = y['Label']
    X_train =pd.read_pickle('train_max_x')
    return y_id,y_lable,X_train

class classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic(self, c, epochs):
        model = LogisticRegression(C=c, dual=False, solver='saga', multi_class='multinomial', max_iter=epochs)
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)

        return preds

    def svm(self, c):
        # n_estimators = 10
        model = LinearSVC(C=c, class_weight='balanced')
        print("start fitting")
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of svm", scores3.mean() * 100)
        print("Score of svm", metrics.accuracy_score(self.y_test, preds))
        return preds

    def KNeighbors(self, iter):
        model = KNeighborsClassifier(n_neighbors=iter, weights='uniform', algorithm='auto', n_jobs=-1)
        model.fit(self.x_train, self.y_train)
        predict = model.predict(self.x_test)
        print(" kneighbors Regression : accurancy_is", metrics.accuracy_score(self.y_test, predict))

class Feature_Processer:
    def split(self, features_set, target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1 - ratio)
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    '''For every image, 
        find a process way
        and split the data set into x train and y train
        then for every feature of image
            consider another layer of segmentation
            consider a layer of CNN 
            output the value'''

    '''More thing can to do, finding the different extraction way
        play with different NN model
        hyper-parameter testing'''



# data stucture is every index is a picture total(50000, 128, 128)
#in total



#X_train[0:1].show()