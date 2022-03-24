"""Trainer

    Train all your model here.
"""
import torch
import os
import sys
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from model import ResNet50

from utils.metric import mean_class_recall

from dataset import Skin7

from losses import NCELoss
# from loss import class_balanced_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
transform = transforms.Compose([
    transforms.RandomCrop(size=[112, 112]),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=[0, 360]),
    transforms.ToTensor(),
])

train_transform = transform  # None
test_transform = transform  # None

trainset = Skin7(train=True, transform=train_transform, target_transform=None)
testset = Skin7(train=False, transform=test_transform, target_transform=None)

net = ResNet50()  # None

batch_size = 24
num_workers = 4
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=num_workers)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=num_workers)

# Loss
nce = NCELoss().to(device)

criterion = nn.CrossEntropyLoss().to(device)

# Optmizer
Lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=Lr)  # None
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

max_epoch = 50
use_cuda = True  # added


def train(model, trainloader):
    model = model.to(device)
    model.train()
    for epoch in range(1, max_epoch + 1):
        running_loss = 0.0
        running_correct = 0
        print(" -- Epoch {}/{}".format(epoch + 2, max_epoch + 1))  # 1

        # model.train()
        # for batch_idx, ([data,
        #                  data_aug], target) in tqdm(enumerate(trainloader)):
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
        # for data, target in trainloader:
        #     print(data, target)
            
            # set all gradients to zero
            optimizer.zero_grad()

            # fetch data
            images, labels = data, target
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # normalization
            images = (images - images.mean()) / (images.std() + 1e-8)

            # output forward
            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, labels)

            # backward and optimize parameters
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            running_correct += torch.sum(pred == labels)
            # pass


def test(model, testloader, epoch):
    model.eval()

    y_true = []
    y_pred = []
    # for _, ([data, data_aug], target) in enumerate(testloader):
    for _, (data, target) in enumerate(testloader):
        # fetch data
        images, labels = data, target
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        # model forward
        outputs = model(images)

        # record the correct
        y_pred = torch.argmax(outputs, dim=1)
        y_true = labels
        # y_pred = torch.augmax(outputs, dim=1)
        # y_true += torch.sum(y_pred == labels)
        # y_true.append(y_pred if y_pred == labels else '') ##
        # pass

    acc = accuracy_score(y_true, torch.Tensor.tolist(y_pred))
    print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
