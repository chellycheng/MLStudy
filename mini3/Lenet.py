import torch.nn as nn
from collections import OrderedDict

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
            ('c3', nn.Conv2d(64,128, kernel_size=(5, 5))),
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
