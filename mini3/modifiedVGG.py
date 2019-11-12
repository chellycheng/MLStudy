# This Python 3 environment comes with many helpful analytics libraries installedfrom PIL import Image
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
# from keras.utils import to_categorical
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, resnet34
from torchvision.models.vgg import VGG, make_layers
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

# from keras.preprocessing.image import ImageDataGenerator

model_urls = {

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',

    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',

    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',

    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

}



# dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lable, transform):
        if lable == 'train':
            self.images = pd.read_pickle('../input/modified-mnist/train_max_x')
            self.images = self.images
            np.place(self.images,self.images<180,0)
            self.labels = [x for [y, x] in pd.read_csv('../input/modified-mnist/train_max_y.csv').to_numpy()]

        elif lable == 'test':
            self.images = pd.read_pickle('../input/modified-mnist/test_max_x')
            self.images = self.images
            np.place(self.images, self.images < 180, 0)
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
            image /= 255

        return image, label


def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# input layer takes single-channel image
# set the number of classes to 10
# used softmax function at the very end of the forward pass in order to have easy to interpret output from the network
class MnistVGGNet(VGG):
    def __init__(self):
        super(MnistVGGNet, self).__init__(make_layers(cfgs['A'], batch_norm=True), num_classes=10)
        # VGG11 'A'; VGG13'B', VGG16'D', VGG19'E'
        # batch_norm: true or false

    def forward(self, x):
        model = torch.softmax(super(MnistVGGNet, self).forward(x), dim=-1)

        return model


tf = Compose([Resize((224, 224)), ToTensor()])
train_set = MyDataset("train", transform=tf)
# image torch size([1, 224, 224])


batch_size = 128
train_portion = 0.8
num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(train_portion * num_train))

train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
                          )
val_loader = DataLoader(train_set,
                        batch_size=batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
                        )

print('done')

start_ts = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 128

model = MnistVGGNet().to(device)

losses = []
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), rho=0.95)
# optim.Adadelta(model.parameters())
# optim.Adadelta(model.parameters(),  rho=0.9, eps=1e-06, weight_decay=1e-6)
# torch.optim.Adam(model.parameters(), weight_decay=1e-5)
# eps
# (params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

batches = len(train_loader)
val_batches = len(val_loader)

# training loop + eval loop
for epoch in range(epochs):
    total_loss = 0
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
    model.train()

    for i, data in enumerate(train_loader):
        X, y = data[0].to(device), data[1].to(device)
        # X, y = data[0], data[1]
        model.zero_grad()
        outputs = model(X)

        loss = loss_function(outputs, y)

        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        total_loss += current_loss
        progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

    torch.cuda.empty_cache()

    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    model.eval()

    total_correct = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)
            outputs = model(X)
            val_losses += loss_function(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1]

            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )

    print(
        f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss / batches)
print(losses)
# print(f"Training time: {time.time() - start_ts}s")

