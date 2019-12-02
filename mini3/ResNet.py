

from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from torchvision.models.resnet import ResNet, BasicBlock
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchsummary import summary

import hiddenlayer as hl
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from keras.preprocessing.image import ImageDataGenerator


# dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lable, transform):
        if lable == 'train':
            self.images = pd.read_pickle('train_max_x').reshape(50000, 1, 128, 128)
            # tmp = pd.read_csv('../input/modified-mnist/train_max_y.csv', encoding='utf-8')
            # tmp = to_categorical((tmp['Label']), num_classes=10, dtype='long')
            datagen.fit(self.images)
            print(self.images.shape)

            self.labels = [x for [y, x] in pd.read_csv('train_max_y.csv').to_numpy()]
        elif lable == 'test':
            self.images = pd.read_pickle('test_max_x').reshape(10000, 1, 128, 128)
            self.labels = None
            #datagen.fit(self.images)

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



# input layer takes single-channel image
# set the number of classes to 10
#used softmax function at the very end of the forward pass in order to have easy to interpret output from the network
class MnistResNet (ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)

    def forward(self, x):
        return torch.softmax( super(MnistResNet, self).forward(x), dim=-1)

# training loop + eval loop

'''



tf = Compose([Resize((224, 224)), ToTensor()])
train_set = MyDataset("train", transform=tf)
# image torch size([1, 224, 224])

batch_size = 256
train_portion = 0.8
lr = 0.05
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

start_ts = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 50

model = MnistResNet().to(device)

losses = []
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())
# optim.Adadelta(model.parameters())

batches = len(train_loader)
val_batches = len(val_loader)


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
print(losses)'''
def sum():
    model = MnistResNet ()
    summary(model, input_size=(1, 128, 128))
    graph= hl.build_graph(model, torch.zeros([1, 1, 128, 128]))
    graph.save("LeNet5")

if __name__ == '__main__':
    sum()


# print(f"Training time: {time.time() - start_ts}s")