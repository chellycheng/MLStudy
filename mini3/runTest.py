%matplotlib inline
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt
import cv2

# Any results you write to the current directory are saved as output.
train_label = 'train_max_y.csv'
train_data = 'train_max_x'
test_data = 'test_max_x'
epochs = 200
batch_size = 256
train_portion = 0.8
lr = 0.05
class transformer:
    #given a picture set, this function can return a transformed picture set and ready for NN
    #glb or nml
    def __init__(self,picture_set,lable):
        self.data = picture_set
        self.lable = lable
    def glb_center(self):
        glb_center = transforms.Compose([transforms.RandomCrop(128,padding=4),
                                         transforms.ToTensor(),transforms.Normalize(104.43527,56.309097)])
        new_data = glb_center.transforms(self.data)
        return new_data
    def normalize(self):
        new_data = self.data/255
        normalizer = transforms.Compose([transforms.RandomCrop(128,padding=4),
                                         transforms.ToTensor()])
        new_data = normalizer.transforms(new_data)
        return new_data
    def run(self):
        if self.lable =="glb":
            result = self.glb_center()
            return result
        elif self.lable =="nml":
            result = self.normalize()
            return result
        else:
            print("Error in lable")

    #global_positive_centering
    #channel_centering
    #local_centering

class dataset:
    def read_dataset(self,lable_y,lable_x,lable_mystry,lable):
        y_train = pd.read_csv(lable_y, encoding='utf-8')
        y_id = y_train['Id']
        y_lable = y_train['Label']
        X_train = pd.read_pickle(lable_x)
        X_test = pd.read_pickle(lable_mystry)
        #X_train(training X), y_lable(training y),X_test(testing x),y_id(some other information)
        #transformer
        X_train = transformer(X_train,lable).run()
        X_test = transformer(X_train,lable).run()
        return X_train, y_lable,X_test,y_id
class ploter:
    def plotAtIndex(self,X_trian,index):
        plt.gray()
        plt.imshow(X_trian[index])
        plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net) .__init__()
        self.conv1 = nn.Conv2d()