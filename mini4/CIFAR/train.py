# Data loading code
import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from dataset import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import FocalLoss







################################
def train(train_loader, model, criterion, optimizer):

    # switch to train mode
    total_loss = 0
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input = input.to(device)
        target = target.to(device)
        model.zero_grad()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print("loss: %s" %(total_loss/(i+1)))
    torch.cuda.empty_cache()
###############################################
def validate (val_loader, model, criterion, epoch, flag='val'):
    val_losses = 0
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            val_losses += loss

            # measure accuracy and record loss
            acc = accuracy(output, target)
            #losses.update(loss.item(), input.size(0))
            #top1.update(acc1[0], input.size(0))
            #top5.update(acc5[0], input.size(0))

            pred = torch.max(output, 1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            print ("\nepoch" + epoch)
            print ("accuracy:" + acc)


        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(out_cls_acc)

    return acc
##############################################

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 'exp', imb_factor=0.01, rand_number=0,
train_dataset = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=0.01,
                                 rand_number=0, train=True,
                                 transform=transform_train)

val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_val)


print(len(train_dataset))  #20431
print(len(val_dataset))  #10000

cls_num_list = train_dataset.get_cls_num_list()
print('cls num list:')
print(cls_num_list)
#args.cls_num_list = cls_num_list


#define train rule
train_rule = 'None'
train_sampler = None
if train_rule == 'None':
    train_sampler = None
    per_cls_weights = None
elif train_rule == 'Resample':
    train_sampler = ImbalancedDatasetSampler(train_dataset)
    per_cls_weights = None
elif train_rule == 'Reweight':
    train_sampler = None
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights)
else:
    warnings.warn('Sample rule is not listed')


# define loss function: focal loss, cross entropy
loss_type = 'CE'
criterion = None
if loss_type == 'CE':
    criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
elif loss_type == 'Focal':
    criterion = FocalLoss(weight=per_cls_weights, gamma=1)
else:
    warnings.warn('Loss type is not listed')


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=(train_sampler is None),
    num_workers=2, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=100, shuffle=False,
    num_workers=2, pin_memory=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet32(num_classes=len(cls_num_list), use_norm=True).to(device)
optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.1,
                                weight_decay=1e-4)

losses = []
epochs = 200

start_ts = time.time()
batches = len(train_loader)
val_batches = len(val_loader)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = lr * epoch / 5
    elif epoch > 180:
        lr = lr * 0.0001
    elif epoch > 160:
        lr = lr * 0.01
    else:
        lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(0, epochs+1):
    adjust_learning_rate(optimizer, epoch, 0.1)
    train(train_loader, model, criterion, optimizer)
    acc1 = validate(val_loader, model, criterion, epoch)






