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

from tqdm import tqdm
import torch.nn.init as init
from dataset import Cholec80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10
use_cuda = True  # added

sequence_length = 3
learning_rate = 5e-4
loss_layer = nn.CrossEntropyLoss()


def train(model, train_loader, learning_rate):

    # print(learning_rate)
    for epoch in range(1, epochs + 1):
        if epoch % 2 == 0:
            learning_rate = learning_rate * 0.5
            optimizer = optim.Adam(model.parameters(),
                                   learning_rate,
                                   weight_decay=1e-5)
        model.train()

        correct = 0
        total = 0
        loss_item = 0

        for data in tqdm(train_loader):

            images, labels = data.to(device), labels.to(device)

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

            ## your code
            pass


def test(model, test_loader):
    print('Testing...')
    model.eval()
    correct = 0
    total = 0
    loss_item = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            pass
    print('Test: Acc {}, Loss {}'.format(correct / total, loss_item / total))
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
    train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, learning_rate=learning_rate)
