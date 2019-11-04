import matplotlib
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
from keras.utils import to_categorical

# Any results you write to the current directory are saved as output.
train_label = '/kaggle/input/modified-mnist/train_max_y.csv'
train_data = '/kaggle/input/modified-mnist/train_max_x'
test_data = '/kaggle/input/modified-mnist/test_max_x'
batch_size = 256
epochs = 50
train_portion = 0.8
lr = 0.05


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lable, transform):
        if lable == 'train':
            self.images = pd.read_pickle(train_data).reshape(50000, 1, 128, 128)
            # tmp = pd.read_csv(train_label, encoding='utf-8')
            # tmp = to_categorical((tmp['Label']), num_classes=10,dtype = 'long')
            # dtype='float32'

            # self.labels = tmp
            self.labels = [x for [y, x] in pd.read_csv(train_label).to_numpy()]
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
            image -= image.mean()
            image /= image.std()

        image = image.reshape(1, 128, 128)
        return image, label


class Net(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 10, kernel_size=5, stride=1),  # input shape(1*128*128),(128-5)/1+1=124
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 20, 5, stride=1, padding=0),  # (62-5)/1+1=58
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)  # the input of full connection
        )
        self.fc = nn.Sequential(  # full connection layers.
            nn.Linear(20 * 29 * 29, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)  # out shape(batch,16,5,5)
        out = out.view(out.size(0), -1)  # out shape(batch,400)
        out = self.fc(out)  # out shape(batch,10)
        return out

class LeNet5(nn.Module):
    """
    Use floor +1
    Input - 1x128x128
    C1 - 64@124x124 (5x5 kernel)
    tanh
    S2 - 64@62x62 (2x2 kernel, stride 2) Subsampling
    C3 - 128@58x58 (5x5 kernel, complicated shit)
    tanh
    S4 - 128@29x29 (2x2 kernel, stride 2) Subsampling
    C5 - 256@25x25 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 32, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(32,128, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(128, 256, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            #not sure why it is just 84
            ('f6', nn.Linear(256*25*25, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

def evaluation(valid_dataloader,net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_dataloader:
            images, labels = data
            images=images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
    return acc

def train():
    # net = Net(1,10)
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

    net = LeNet5()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
    print('Start Training')
    epochs = 50
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (pic, label) in enumerate(train_dataloader):
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
            if i % 120 == 119:  # print every 120 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 119))
                running_loss = 0.0
                evaluation(valid_dataloader,net)
    print('Finished Training')

    return net
def perdict(test_dataloader,net):
    with torch.no_grad():
        result = []
        for data in test_dataloader:
            #set to GPU
            data=data.cuda()
            #predict by using the model
            outputs = net(data)
            # prediction
            _, predicted = torch.max(outputs.data, 1)
            result.append(predicted)
    with open('result.csv', 'w') as f:
            for item in result:
                f.write("%s\n" % item)
    #if result is out, need to add index number manually


if __name__ == '__main__':

    train()
