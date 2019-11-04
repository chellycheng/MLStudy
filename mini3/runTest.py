import matplotlib
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import to_categorical

# Any results you write to the current directory are saved as output.
train_label = 'train_max_y.csv'
train_data = 'train_max_x'
test_data = 'test_max_x'
epochs = 50
batch_size = 256
train_portion = 0.8
lr = 0.05


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,lable , transform):
        if lable == 'train':
            self.images = pd.read_pickle(train_data).reshape(50000, 1,128, 128)
            tmp = pd.read_csv(train_label, encoding='utf-8')
            tmp = to_categorical((tmp['Label']), num_classes=10, dtype='float32')
            self.labels = tmp
        elif lable == 'test':
            self.images = pd.read_pickle(test_data).reshape(50000,1, 128, 128)
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
            image -= image.mean()
            image /= image.std()

        image = image.reshape(1, 128, 128)
        return image, label


class Net(nn.Module):
    def __init__(self,in_dim,n_class):
        super(Net,self) .__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 10, kernel_size=5, stride=1),# input shape(1*128*128),(128-5)/1+1=124
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, 5, stride=1, padding=0),  # (62-5)/1+1=58
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # the input of full connection
        )
        self.fc = nn.Sequential(  # full connection layers.
            nn.Linear(20*29*29, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out

def train():


    tf = transforms.Compose([transforms.RandomCrop(128, padding=4),
                             transforms.ToTensor(),
                            # transforms.Normalize(104.43527, 56.309097)
                             ])
    train_set = MyDataset("train", transform=tf)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    # Instantiate dataloader
    train_dataloader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                  num_workers=4)
    valid_dataloader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
                                  num_workers=4)
    net = Net(1, 10)
    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    print('Start Training')
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (pic, label)in enumerate(train_dataloader):
            label = label.cuda()
            pic = pic.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(pic)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 256 == 255:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 255))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    train()