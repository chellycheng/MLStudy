import pandas as pd
import numpy as np
import matplotlib as plt
from scipy.special import expit
from sklearn.metrics import confusion_matrix
from statistics import mean
# one can add the limit number of iterations
class logistics_reg:
    def __init__(self ,x_train,y_train, x_test,y_test,parameter):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.w = parameter

    def fit(self,iteration, studyRate, use_reg):
        for it in range(iteration):
            self.update(studyRate,use_reg)
        return self.w

    def update(self,studyRate,use_reg):
        prediction = 0
        for i in range(self.x_train.shape[0]):
            x_i = self.x_train[i].T
            prediction += x_i.dot(self.y_train[i]-expit(self.w.T.dot(x_i)))
        if use_reg[0]:
            lmd = use_reg[1]
            #updata every theta except the w0
            w = self.w
            w[0] = 0
            prediction += (2*lmd)*w
            # aka L2-regularization
        modifying_number = studyRate * prediction
        self.w = self.w + modifying_number

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        y_per =expit(X.dot(self.w))
        y_hat = np.where(y_per > 0.5, 1, 0)
        return y_hat

    def evaluate_acc(self):
        predicted_classes = self.predict(self.x_test)
        predicted_classes = predicted_classes.flatten()
        n = len(predicted_classes)
        count = 0
        for i in range(n):
            if predicted_classes[i] == self.y_test[i]:
                count +=1
        accuracy = float(count)/n
        return accuracy

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.predict(self.x_test))
        print("The confusion matrix is\n",cm)
        ture_positive =cm[0,0]
        ture_negative =cm[1,1]
        false_positive =cm[0,1]
        false_negative =cm[1,0]
        print("Recall", ture_positive/(ture_positive+false_negative))
        print("Precision",ture_positive/(ture_positive+false_positive))

    def validationSets(self,fold_num):
        n = len(self.x_train)
        fsize = int(n/fold_num)
        remain=n%fold_num
        folds_x=[]
        folds_y=[]
        start=0
        for k in range(fold_num):
            if k < remain:
                end=start+fsize+1
                folds_x.append(self.x_train[start:end])
                folds_y.append(self.y_train[start:end])
                start=end
            else:
                end=start+fsize
                folds_x.append(self.x_train[start:end])
                folds_y.append(self.y_train[start:end])
                start=end
        #print(folds_x,folds_y)
        return folds_x,folds_y

    def crossValidation(self,folds_x,folds_y, iter, study_rate, w,use_reg):
        # iter=gradient iteration times
        accurancy = float(0.0000000)
        k = len(folds_x)
        for i in range(k):
            valid_x = folds_x[i]
            valid_x = np.array(valid_x)

            valid_y = folds_y[i]
            valid_y = np.array(valid_y)

            train_x = [folds_x[j] for j in range(k) if j != i]
            train_x = np.concatenate(train_x, axis=0)
            train_x = np.array(train_x)

            train_y = [folds_y[j] for j in range(k) if j != i]
            train_y = np.concatenate(train_y, axis=0)
            train_y = np.array(train_y)

            X = logistic_reg(train_x, train_y, valid_x, valid_y, w)
            X.fit(iter, study_rate,use_reg)
            accurancy+=X.evaluate_acc()
        accurancy = accurancy/ k
        print("Corss Validation Accurancy is ", accurancy)

class lda_model:
    def __init__(self):
        self.w0 = 0.00000000
        self.u0 = 0.00000000
        self.u1 = 0.00000000
        self.cov = 0.00000000

    def fit(self, featuresDataSet, classDataSet):
        X = np.copy(featuresDataSet)
        Y = np.copy(classDataSet)

        p1 = (np.count_nonzero(Y == 1)) / float(len(Y))
        p0 = (len(Y) - np.count_nonzero(Y == 1)) / float(len(Y))

        X0 = np.zeros(shape=(X.shape[0], X.shape[1]))
        X1 = np.zeros(shape=(X.shape[0], X.shape[1]))
        m = 0
        n = 0
        for j in range(len(Y)):
            if Y[j] == 0:
                X0[m] = X[j]
                m = m + 1
            else:
                X1[n] = X[j]
                n = n + 1
        self.u0 = X0.mean(0) * (X.shape[0]) / (X.shape[0] - np.count_nonzero(Y == 1))
        self.u1 = X1.mean(0) * (X.shape[0]) / (np.count_nonzero(Y == 1))

        self.cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for k in range(len(Y)):
            if Y[k] == 0:
                self.cov = self.cov + (np.matmul(np.transpose([X[k] - self.u0]), [X[k] - self.u0]))
            else:
                self.cov = self.cov + (np.matmul(np.transpose([X[k] - self.u1]), [X[k] - self.u1]))

        self.cov = (self.cov) / (len(Y) - 2)
        self.w0 = np.log(p1 / p0) - (1 / 2) * (np.matmul(np.matmul(self.u1, np.linalg.inv(self.cov)), self.u1)) + (1 / 2) * (np.matmul(np.matmul(self.u0, np.linalg.inv(self.cov)), self.u0))

    def predictOneExample(self, dataX):
        value = self.w0 + np.matmul(np.matmul(dataX, np.linalg.inv(self.cov)), self.u1 - self.u0)
        if value > 0:
            return 1
        else:
            return 0

    def predict(self, pX):
        values = np.zeros(shape=(pX.shape[0]))
        for g in range(pX.shape[0]):
            values[g] = self.predictOneExample(pX[g])
        return values

    def evaluate_acc(self, trueY, eY):
        count = 0
        index = 0
        for g in range(len(eY)):
            if eY[g] == trueY[g]:
                count = count + 1
                index = index + 1
            else:
                index = index + 1
        return count / index

    def confusion_matrix(self,x_test,y_test):
        cm = confusion_matrix(y_test, self.predict(x_test))
        print("The confusion matrix is\n",cm)
        ture_positive =cm[0,0]
        ture_negative =cm[1,1]
        false_positive =cm[0,1]
        false_negative =cm[1,0]
        print("Recall", ture_positive/(ture_positive+false_negative))
        print("Precision",ture_positive/(ture_positive+false_positive))

    def validationSets(self,fold_num,x_train,y_train):
        n = len(x_train)
        fsize = int(n/fold_num)
        remain=n%fold_num
        folds_x=[]
        folds_y=[]
        start=0
        for k in range(fold_num):
            if k < remain:
                end=start+fsize+1
                folds_x.append(x_train[start:end])
                folds_y.append(y_train[start:end])
                start=end
            else:
                end=start+fsize
                folds_x.append(x_train[start:end])
                folds_y.append(y_train[start:end])
                start=end
        return folds_x,folds_y

    def crossValidation(self,folds_x,folds_y):
        # iter=gradient iteration times
        accurancy = float(0.0000000)
        k = len(folds_x)
        for i in range(k):
            valid_x = folds_x[i]
            valid_x = np.array(valid_x)

            valid_y = folds_y[i]
            valid_y = np.array(valid_y)

            train_x = [folds_x[j] for j in range(k) if j != i]
            train_x = np.concatenate(train_x, axis=0)
            train_x = np.array(train_x)

            train_y = [folds_y[j] for j in range(k) if j != i]
            train_y = np.concatenate(train_y, axis=0)
            train_y = np.array(train_y)
            lda = lda_model()
            lda.fit(train_x, train_y)
            accurancy+=lda.evaluate_acc(valid_y,lda.predict(valid_x))
        accurancy = accurancy/ k
        print("Corss Validation Accurancy is ", accurancy)

class help_method():
    def zeroOrOne(x):
        if x > 5:
            return 1
        else:
            return 0

    def train_test_x_y(feature_set,r,y_name):
        sp = int(len(feature_set) * r)
        train = feature_set[sp:]
        test = feature_set[:sp]
        return train.drop(y_name,axis=1).to_numpy(), train[y_name].to_numpy(), test.drop(y_name,axis=1).to_numpy(), test[y_name].to_numpy()

    def shuffle(data_frame):
        return data_frame.sample(frac=1)

    def random_parameter_generator(length):
        # w = [random.randint(-1, 1)] * length
        w = [0] * length
        return np.asarray(w)

    def dummy_insert(data):
        data.insert(0, "dummy", 1)

    def take_mean(twod_array):
        m = len(twod_array[0])
        n = len(twod_array)
        b=[]
        for i in range(n):
            b.append((sum(twod_array[i]))/m)
        b=np.asarray(b)
        print(b.shape)
        b=b.reshape((n,1))
        return b

class reader_processer:

    def read_process_wine(name1,is_drop):
        data = pd.read_csv('winequality-red.csv', ';')
        data['quality'] = np.where(data['quality'] > 5, 1, 0)
        if is_drop:
            for s in name1:
                data=data.drop(s,axis=1)
        #wine_data_shuffle = help_method.shuffle(data)
        return data

    def read_process_cancer(name1,is_drop):
        data = pd.read_csv('breast-cancer-wisconsin.data', ',',
                            names=["ID number", "Clump Thickness", "Uniformity of cell size",
                                   "Uniformity of cell shape",
                                   "Marginal adhesion", "Single epithelial cell size", "Bare nuclei", "Bland chromatin",
                                   "Normal nucleoli", "Mitoses", "Class"])
        data = data[(data.astype(str) != '?').all(axis=1)]
        data['Bare nuclei']=pd.to_numeric(data['Bare nuclei'])
        data = data.drop('ID number', axis=1)
        data['Class'] = np.where(data['Class'] > 3, 1, 0)

        if is_drop:
            for s in name1:
                data=data.drop(s,axis=1)

        cancer_data_shuffle = help_method.shuffle(data)
        return cancer_data_shuffle

class experiment:

    def wine_experiment_logistic(drop_feature,is_drop,ratio,iter,a,use_reg):
        wine_data_shuffle = reader_processer.read_process_wine(drop_feature,is_drop)
        help_method.dummy_insert(wine_data_shuffle)
        train_x, train_y, test_x, test_y = help_method.train_test_x_y(wine_data_shuffle, ratio,'quality')
        w = help_method.random_parameter_generator(train_x.shape[1])
        for i in a:
            model = logistic_reg(train_x, train_y, test_x, test_y, w)
            model.fit(iter, i, use_reg)
            print("Study Rate at", i)
            folds_x, folds_y = model.validationSets(5)
            model.crossValidation(folds_x, folds_y, iter, i, w,use_reg)
            print("Logistic Accurancy:",model.evaluate_acc())
            model.confusion_matrix()

    def cancer_experiment_logistic(drop_feature,is_drop,ratio,iter,a,use_reg):
        cancer_data_shuffle = reader_processer.read_process_cancer(drop_feature,is_drop)
        help_method.dummy_insert(cancer_data_shuffle)
        train_x, train_y, test_x, test_y = help_method.train_test_x_y(cancer_data_shuffle, ratio, 'Class')
        w = help_method.random_parameter_generator(train_x.shape[1])

        for i in a:
            model = logistic_reg(train_x, train_y, test_x, test_y, w)
            model.fit(iter, i, use_reg)
            folds_x, folds_y = model.validationSets(5)
            print("Study Rate at", i)
            model.crossValidation(folds_x, folds_y, iter, i, w,use_reg)
            print("Linear Model Accurancy:", model.evaluate_acc())
            model.confusion_matrix()

    def wine_experiment_LDA(drop_feature,is_drop,ratio):
        wine_data_shuffle = reader_processer.read_process_wine(drop_feature,is_drop)
        train_x, train_y, test_x, test_y = help_method.train_test_x_y(wine_data_shuffle, ratio, 'quality')

        lda = lda_model()
        lda.fit(train_x, train_y)
        folds_x, folds_y = lda.validationSets(5, train_x, train_y)
        lda.crossValidation(folds_x, folds_y)
        print("LDA accurancy", lda.evaluate_acc(test_y, lda.predict(test_x)))
        lda.confusion_matrix(test_x,test_y)

    def cancer_experiment_LDA(drop_feature,is_drop,ratio):
        cancer_data_shuffle = reader_processer.read_process_cancer(drop_feature,is_drop)
        train_x, train_y, test_x, test_y = help_method.train_test_x_y(cancer_data_shuffle, ratio, 'Class')

        lda = lda_model()
        lda.fit(train_x, train_y)
        folds_x, folds_y = lda.validationSets(5,train_x,train_y)
        lda.crossValidation(folds_x, folds_y)
        print("LDA accurancy", lda.evaluate_acc(test_y, lda.predict(test_x)))
        lda.confusion_matrix(test_x,test_y)


def main():
    name1 = ["pH", "residual sugar", "free sulfur dioxide", "fixed acidity"]
    name2 = name1 + ["chlorides"]
    name3 = name2 + ["citric acid"]
    name4 = name3 + ["density"]
    name5 = name4 + []
    names = [name1, name2, name3, name4]
    """""
    #experiment.cancer_experiment()
    name1 = ["pH", "residual sugar", "free sulfur dioxide", "fixed acidity"]
    name2 = name1+["chlorides"]
    name3 = name2+["citric acid"]
    name4 = name3+["density"]
    names=[name1,name2,name3,name4]
    for nm in names:
        experiment.wine_experiment(nm,True)
    """""
    ratio = 0.1
    iter = 100
    use_reg = [False, 0.005]
    a = [ 0.0001, 0.001]
    print("Wine set experiment")
    #experiment.wine_experiment_LDA(name1,True,ratio)
    print("-----------------\n")
    experiment.wine_experiment_logistic(name1,False,ratio,iter,a,use_reg)
    print("\nCancer set experiment")
    #experiment.cancer_experiment_LDA(name1,False,ratio)
    print("-----------------\n")
    #experiment.cancer_experiment_logistic(name1,False,ratio,iter,a,use_reg)

if __name__ == '__main__':
    main()
