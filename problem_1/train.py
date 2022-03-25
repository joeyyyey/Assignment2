from numpy import ndarray
from dataset import LAHeart
# import model
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from augmentation import RandomCrop, RandomRotFlip, ToTensor

from model import UNet  # RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from loss import DiceLoss

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

if __name__ == '__main__':
    max_epoch = 1000
    batch_size = 2
    use_cuda = True  # added

    # model initialization
    model = UNet  # None

    train_dst = LAHeart(split='train', transform=transform)
    # test_dst = LAHeart(split='test', transform=None)  # commented out

    # Dataloader
    train_loader = DataLoader(train_dst,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # test_loader = DataLoader(test_dst,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          num_workers=4,
    #                          pin_memory=True)

    if use_cuda:
        model = model.cuda()

    # loss function
    # criterion = nn.CrossEntropyLoss()
#     criterion = DiceLoss()

    # optimizer
    Lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=Lr)  # None
    # optimizer = torch.optim.Adam(model.parameters(),lr=Lr)

    loss_list = []  # added
    acc_train_list = []  # added

    for epoch in range(max_epoch):
        running_loss = 0.0
        running_correct = 0
        print(" -- Epoch {}/{}".format(epoch + 1, max_epoch))

        model.train()
        for batch in train_loader:
            # set all gradients to zero
            optimizer.zero_grad()

            # fetch data
            images, labels = batch['image'], batch['label']
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            
            # normalization
            images = (images - images.mean())/(images.std() + 1e-8)

            # model forward
            outputs = model(images)

            # calculate loss
            loss_seg = F.cross_entropy(outputs, labels)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
            loss = 0.5*(loss_seg+loss_seg_dice)
#             loss = criterion(outputs, labels)

            # backward and optimize parameters
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            running_correct += torch.sum(pred == labels)

        # record loss, accuracy
        loss = running_loss / len(train_dst)
        loss_list.append(loss)
        acc_train = running_correct / len(train_dst)
        acc_train_list.append(acc_train.item())

        # pass
