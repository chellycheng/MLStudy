import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from matplotlib import pyplot as plt
import pickle
# Any results you write to the current directory are saved as output.
train_label = 'train_max_y.csv'
train_data = 'train_max_x'
test_data = 'test_max_x'
''' Extends the Dataset'''
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,lable , transform):
        if lable == 'train':
            self.images = pd.read_pickle(train_data).reshape(50000, 1, 128, 128)
            tmp = pd.read_csv(train_label, encoding='utf-8')
            self.labels = tmp['Label']
        elif lable == 'test':
            self.images = pd.read_pickle(test_data).reshape(50000, 1, 128, 128)
            self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = -1

        image = self.images[index]
        image = Image.fromarray(image.reshape(128, 128))

        if self.labels is not None:
            label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)
            #image -= image.mean()
            #image /= image.std()

        image = image.reshape(1, 128, 128)
        return image, label

if __name__ == '__main__':
    tf = transforms.Compose([transforms.RandomCrop(128, padding=4),
                             transforms.ToTensor(), transforms.Normalize(104.43527, 56.309097)])
    train_set = MyDataset("train", transform=tf)

    #with open('Processed_data.clf', 'wb') as output:
    #    pickle.dump(train_set, output)