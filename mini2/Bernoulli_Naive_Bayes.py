# import some library
# -*- coding: utf-8 -*
import numpy as np
import string
import pandas as pd
#model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
#help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#help_feature_process
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams
#help_analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#analysis
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statistics import mean

#read from the file, possible write for testing
class Reader:

    def read(self,path):
        data = pd.read_csv(path, encoding="utf-8")
        return data

    def shuffle(self,df):
        df.shuffle()

    def write(self,name):
        f = open(name, "w")
        for line in self.data:
            f.write(line)
        f.close()

    def extractColToString(self,df,col_name):
        data_p = [ (line) for line in self.file]
        return data_p

# 2 Cleaning of the data
class Cleaner:
    def __init__(self, sample_list,use_lemmer,use_stemmer, use_stopwords):
        self.sents_list = sample_list
        self.words_list = [self.splitter(w) for w in sample_list]
        self.s =use_stemmer
        self.l =use_lemmer
        self.st = use_stopwords

    def splitter(self,sample_list):
        pos_words = sample_list.split()
        return pos_words

    def remove_punc(self):
        removed_punc = []
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            removed_punc.append( [w.translate(table) for w in s] )
        self.words_list = removed_punc

    def lowercase(self):
        lowered = []
        for s in self.words_list:
            lowered.append( [w.lower() for w in s])
        self.words_list = lowered

    def remove_noncharacter(self):
        remove_nonchar = []
        for s in self.words_list:
            remove_nonchar.append([w for w in s if w.isalnum()])
        self.words_list = remove_nonchar

    def remove_stopWord(self):
        removed_stop = []
        stop_words = stopwords.words('english')
        for s in self.words_list:
            removed_stop.append([w for w in s if not w in stop_words])
        self.words_list = removed_stop

    def lemmatizer(self):
        lemmatized = []
        lemmatizer = WordNetLemmatizer()
        for s in self.words_list:
            lemmatized.append([lemmatizer.lemmatize(w) for w in s])
        self.words_list = lemmatized

    def stemmer(self):
        stemmed = []
        porter = PorterStemmer()
        for s in self.words_list:
            stemmed.append( [porter.stem(word) for word in s])
        self.words_list = stemmed

    def clean_low_puc_nc_le_stop(self):
        cleaned = []
        #porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            cleaned.append([lemmatizer.lemmatize(word.translate(table).lower()) for word in s if word not in stop_words])
        self.words_list = cleaned

    def cleaned(self):
        self.lowercase()
        self.remove_punc()
        self.remove_noncharacter()
        if self.l:
            self.lemmatizer()
        if self.s:
            self.stemmer()
        if self.st:
            self.remove_stopWord()
        result = self.joined()
        return result

    def joined(self):
        sents = []
        for s in self.words_list:
            sents.append(' '.join(s))
        return sents

class Feature_Processer:
    def split(self,features_set,target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1-ratio)
        return X_train, X_test, y_train, y_test
    #n_grams, min_df
    def count_vector_features_produce(self, X_train, X_test, thresold):
        cv = CountVectorizer(binary=True,min_df=thresold)
        cv.fit(X_train)
        X = cv.transform(X_train)
        X_test = cv.transform(X_test)
        return X, X_test

    def tf_idf(self,X_train,X_test,n_grams,thresold):
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=n_grams,min_df =thresold)
        vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
        vectors_test_idf = tf_idf_vectorizer.transform(X_test)
        return vectors_train_idf,vectors_test_idf

    """
    def bigram_extractor(self):
        bigramFeatureVector = []
        for item in bigrams(tweetString.split()):
            bigramFeatureVector.append(' '.join(item))
        return bigramFeatureVector
    """
class classifier:
    def __init__(self, x_train, x_test, y_train,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic(self, c):
        model = LogisticRegression(C=c, dual=False, solver='lbfgs',multi_class= 'multinomial')
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores1 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of Logistic in Cross Validation", scores1.mean() * 100)
        print("Losistic Regression : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n", cm)
        print("Report", classification_report(self.y_test, preds))

    def Ber_NaiveBayes(self, alpha):
        model = BernoulliNB(alpha=alpha).fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores2 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of Naive Bayes", scores2.mean() * 100)
        print("Bernoulli Naive Bayes : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
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

    def decision_tree(self):
        #criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False
        model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)
        model.fit(self.x_train, self.y_train)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of decision tree in Cross Validation", scores3.mean() * 100)
        print("decision tree  : accurancy_is", metrics.accuracy_score(self.y_test, model.predict(self.x_test)))

    def QDA(self):
        model = QuadraticDiscriminantAnalysis()
        model.fit(self.x_train.toarray(), self.y_train)
        scores3 = cross_val_score(model, self.x_train.toarray(), self.y_train, cv=5, scoring='accuracy')
        print("Score of QDA in Cross Validation", scores3.mean() * 100)
        print("QDA Regression : accurancy_is", metrics.accuracy_score(self.y_test, model.predict(self.x_test.toarray())))

    def dummy(self):
        clf = DummyClassifier(strategy='stratified', random_state=0)
        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        print("Random Baseline's accurancy", score)

    def  multNB(self):
        model = MultinomialNB()
        model.fit(self.x_train, self.y_train)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of MultinomialNB in Cross Validation", scores3.mean() * 100)
        print(" MultinomialNB Regression : accurancy_is", metrics.accuracy_score(self.y_test, model.predict(self.x_test)))

def main():
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    #use_lemmer,use_stemmer, use_stopwords
    cleaner_train = Cleaner(data_train,True,False,True)
    cleaner_train.cleaned()

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train,data_test,0.8)
    X_train, X_test = Feature_Processer().tf_idf(X_train, X_test,(1,2),1)

    clf = classifier(X_train, X_test, y_train, y_test)
    #logistic converges deadly
    #clf.logistic(1.0)
    clf.svm(0.2)
    #跑不动
    #clf.decision_tree()
    #clf.QDA()
    clf.multNB()
    #svm approximately 56%
    #multinomial Nb with 54%


if __name__ == "__main__":
    main()