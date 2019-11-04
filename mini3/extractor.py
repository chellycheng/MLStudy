
import sklearn
from skimage.feature import daisy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import numpy as np
import cv2
from skimage import feature, exposure


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

    def svm(self, c):
        model = LinearSVC(C=c, class_weight='balanced')
        print("start fitting")
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of svm", scores3.mean() * 100)
        #print("Score of svm", metrics.accuracy_score(self.y_test, preds))
        #return preds

    def KNeighbors(self, iter):
        model = KNeighborsClassifier(n_neighbors=iter, weights='uniform', algorithm='auto', n_jobs=-1)
        model.fit(self.x_train, self.y_train)
        predict = model.predict(self.x_test)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of svm", scores3.mean() * 100)
        #print(" kneighbors Regression : accurancy_is", metrics.accuracy_score(self.y_test, predict))

class Feature_Processer:
    def split(self, features_set, target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1 - ratio)
        return X_train, X_test, y_train, y_test

    '''
        1. Kaze Features - 测试中
        2. Daisy Features - 测试中
        3. HoG Features - 测试过了建议最终不用，但可以写进测试。
        4. Flatten - 
    '''
    def kaze(self,image, vector_size=32):
        alg = cv2.KAZE_create()
        kps = alg.detect(image)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same sizepip3 install torch torchvision
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return dsc

    def dazy(self,images,test_images):
        #Not finished testing, don't use this one
        daisy_features_train_set = np.zeros((len(images), 104))
        for i in range(len(images)):
            descs, descs_img = daisy(images[i], step=180, radius=20, rings=2, histograms=6,
                                     orientations=8, visualize=True)
            daisy_features_train_set[i] = descs.reshape((1, 104))

            fig, ax = plt.subplots()
            ax.axis('off')
            ax.imshow(descs_img)
            descs_num = descs.shape[0] * descs.shape[1]
            ax.set_title('%i DAISY descriptors extracted:' % descs_num)
            plt.show()

        print(daisy_features_train_set.shape)

        #np.savetxt("train_daisy.csv", daisy_features_train_set, delimiter=",")

        print( "Daisy: Saving features' loop for test")
        daisy_features_test_set = np.zeros((len(test_images), 104))
        for i in range(len(test_images)):
            descs, descs_img = daisy(test_images[i], step=180, radius=20, rings=2, histograms=6,
                                     orientations=8, visualize=True)
            daisy_features_test_set[i] = descs.reshape((1, 104))

        print(daisy_features_test_set.shape)

    def hog(self,image):
        # for single image
        fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(16, 16),
                                    cells_per_block=(2, 2), visualize=True)
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return hog_image_rescaled
    def flatten(self,x_train,x_test):

        # convert to float
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        # normalize
        x_train /= 255
        x_test /= 255
        return x_train, x_test


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

