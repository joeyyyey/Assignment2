from numpy import ndarray
from dataset import LAHeart
# import model
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from augmentation import RandomCrop, RandomRotFlip, ToTensor

from model import UNet  # RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from loss import DiceLoss, dice_loss
from test import test
# import plot
from matplotlib import pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify patch size for cropping
patch_size = (112, 112, 80)

# transform compose
transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.RandomCrop(size=[112,112]),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=[0,360]),
    RandomRotFlip(),
    RandomCrop(patch_size),
    ToTensor()
])

max_epoch = 5
batch_size = 2
use_cuda = True  # added
Lr = 1e-4

# model initialization
model = UNet()  # None

train_dst = LAHeart(split='train', transform=transform)
# test_dst = LAHeart(split='test', transform=None)  # commented out

# Dataloader
train_loader = DataLoader(train_dst,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)


def train(model: UNet, trainloader: DataLoader, use_cuda, max_epoch, learning_rate):

    if use_cuda:
        model = model.cuda()

    # loss function
    # criterion = nn.CrossEntropyLoss()
    # criterion = DiceLoss()
    criterion = torch.nn.BCELoss()

    # optimizer
    # Lr = 0.01
    # optimizer = torch.optim.SGD(model.parameters(), lr=Lr)  # None
    # optimizer = torch.optim.Adam(model.parameters(), lr=Lr)

    loss_list = []  # added
    acc_train_list = []  # added
    acc_train = 0  # added
    total = 0  # added

    for epoch in range(max_epoch):
        if epoch % 2 == 0:
            learning_rate = learning_rate * 0.5
            optimizer = torch.optim.Adam(model.parameters(),
                                   learning_rate,
                                   weight_decay=1e-4)
        running_loss = 0.0
        running_correct = 0
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))

        model.train()
        for batch in trainloader:
            # set all gradients to zero
            optimizer.zero_grad()

            # fetch data
            images, labels = batch['image'], batch['label']
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            # normalization
            images = (images - images.mean()) / (images.std() + 1e-8)

            # model forward
            outputs = model(images)

            # calculate loss
            # loss_seg = F.cross_entropy(outputs, labels)
            # outputs_soft = F.softmax(outputs, dim=1)
            # loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
            # loss = 0.5*(loss_seg+loss_seg_dice)
            loss = criterion(outputs, labels)

            # backward and optimize parameters
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            total += outputs.size(0)
            # running_correct += torch.sum(pred == labels)

        # record loss, accuracy
        loss = running_loss / len(train_dst)
        loss_list.append(loss)
        # running_correct += torch.sum(pred == labels)
        running_correct = (pred == labels).float()
        # acc_train = running_correct / total
        acc_train = running_correct.sum() / running_correct.numel()
        acc_train_list.append(acc_train.item())

        print("Loss {:.4f}, Train Accuracy {:.4f}%".format(
            loss,
            acc_train * 100,
        ))

        # plot(x_start=1, x_end=max_epoch, y1=loss_list, y2=None, title='Training Loss Curve', xlabel='Epoch', ylabel='Loss')
        # print(len(train_dst))
        # print(running_correct)

    x = np.arange(1, max_epoch+1)
    y = loss_list
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, y)
    plt.show()
    # pass


if __name__ == '__main__':
    train(model=model, trainloader=train_loader, use_cuda=use_cuda, max_epoch=max_epoch, learning_rate=Lr)
    test(model=model)
