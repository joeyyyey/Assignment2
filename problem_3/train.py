import random
import shutil
import time
import warnings
# from thop import profile
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm
import torch.nn.init as init
from dataset import Cholec80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epoch = 10
use_cuda = True  # added

sequence_length = 3
learning_rate = 5e-4
loss_layer = nn.CrossEntropyLoss()


def train(model, train_dataloader, learning_rate, traindataset):

    # print(learning_rate)
    for epoch in range(1, max_epoch + 1):
        # running_loss = 0.0
        # running_correct = 0
        print(" -- Epoch {}/{}".format(epoch, max_epoch))
        if epoch % 2 == 0:
            learning_rate = learning_rate * 0.5
            optimizer = optim.Adam(model.parameters(),
                                   learning_rate,
                                   weight_decay=1e-5)
        model.train()

        correct = 0
        total = 0
        loss_item = 0
        loss_list = []
        acc_train_list = []

        for images, labels in tqdm(train_dataloader):

            images, labels = images.to(device), labels.to(device)

            # fetch data
            # images, labels = data
            # if use_cuda:
            #     images = images.cuda()
            #     labels = labels.cuda()
            
            # model forward
            outputs = model(images)

            # set all gradients to zero
            optimizer.zero_grand()

            # normalization
            images = (images - images.mean()) / (images.std() + 1e-8)

            # calculate loss
            loss = loss_layer(outputs, labels)

            # backward and optimize parameters
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            loss_item += loss.item()
            total += outputs.size(0)
            correct += torch.sum(pred == labels)
            ## your code
        
        loss_list.append(loss_item)
        acc_train = correct / len(traindataset)
        acc_train_list.append(acc_train.item())
            #pass
        print('Train: Acc {:.4f}, Loss {:.4f}'.format(acc_train, loss_item))

    x = np.arange(1, max_epoch+1)
    y1 = loss_list
    y2 = acc_train_list
    fig, ax1 = plt.subplots()
    plt.title('Training Curves')
    ax2 = ax1.twinx()
    ax1.plot(x, y1, color = 'blue')
    ax2.plot(x, y2, color = 'orange')
    ax1.set_xlabel('Epoch', color = 'black')
    ax1.set_ylabel('Loss', color = 'blue')


def test(model, test_dataloader, testdataset):
    print('Testing...')
    model.eval()
    correct = 0
    total = 0
    loss_item = 0
    loss_list = []
    acc_test_list = []
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = loss_layer(outputs, labels)

            pred = torch.argmax(outputs, dim=1)
            loss_item += loss.item()
            total += outputs.size(0)
            correct += torch.sum(pred == labels)
            # pass
        loss_list.append(loss_item)
        acc_test = correct / len(testdataset)
        acc_test_list.append(acc_test.item())

    print('Test: Acc {:.4f}, Loss {:.4f}'.format(acc_test, loss_item))
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':

#     train_transform = transforms.Compose([
#         transforms.RandomCrop(size=[112, 112]),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(degrees=[0, 360]),
#         transforms.ToTensor()
# ])
#     # train_transform = transform  # None
#     test_transform = transforms.Compose([
#         # transforms.RandomCrop(size=[112, 112]),
#         # transforms.RandomVerticalFlip(),
#         # transforms.RandomHorizontalFlip(),
#         # transforms.RandomRotation(degrees=[0, 360]),
#         transforms.ToTensor()
#     ])
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True) # None
    learning_rate = 1e-4
    traindataset = Cholec80(train=True, transform=None) # None
    train_dataloader = DataLoader(traindataset,
                                  batch_size=32,
                                  shuffle=True,
                                  drop_last=True)
    testdataset = Cholec80(train=False, transform=None) # None
    test_dataloader = DataLoader(testdataset,
                                 batch_size=32,
                                 shuffle=False,
                                 drop_last=False)
    train(model=model, train_dataloader=train_dataloader, learning_rate=learning_rate, traindataset=traindataset)
    test(model=model, test_dataloader=test_dataloader, testdataset=testdataset)
