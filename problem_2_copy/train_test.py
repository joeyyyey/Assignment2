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
from matplotlib import pyplot as plt
# import InfoNCE

# from model import ResNet50

# from utils.metric import mean_class_recall

from dataset import Skin7

# from losses import NCELoss, InfoNCE, ContrastiveLoss
# from loss import class_balanced_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.486, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

###
train_transform = transforms.Compose([
    transforms.RandomCrop(size=[112, 112]),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=[0, 360]),
    transforms.ToTensor()
    # transforms.Normalize(mean, std)
])

# train_transform = transform  # None
test_transform = transforms.Compose([
    # transforms.RandomCrop(size=[112, 112]),
    transforms.CenterCrop(size=[150, 150]),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=[0, 360]),
    transforms.ToTensor()
    # transforms.Normalize(mean, std)
])

trainset = Skin7(train=True, transform=train_transform, target_transform=None)
testset = Skin7(train=False, transform=test_transform, target_transform=None)

net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True) # ResNet50()  # None

batch_size = 30 # 24
num_workers = 4 # 4
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
# nce = NCELoss(vocab_size=7).to(device)
# nce = InfoNCE().to(device)
# nce = ContrastiveLoss(margin=1.5).to(device)
criterion = nn.CrossEntropyLoss().to(device)

# Optmizer
Lr = 1e-4/3 # 0.01
weight_decay =1e-6
# optimizer = torch.optim.SGD(net.parameters(), lr=Lr)  # None
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
# optimizer = torch.optim.SGD(net.parameters(), lr=Lr, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer, mode='min', factor=0.1, patience=50, verbose=True,
#                 threshold=1e-4)
optimizer = torch.optim.Adam(net.parameters(), lr=Lr,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay= weight_decay,
                           amsgrad=True)

max_epoch = 50 # 50
# use_cuda = True  # added

def train(model, trainloader, testloader, max_epoch, optimizer):
    # model = model.to(device)
    train_loss_list = []
    test_loss_list = []
    # loss_list = [] # added
    acc_train_list = [] # added
    acc_test_list = []

    for epoch in range(1, max_epoch + 1):
        running_loss = 0.0
        running_correct = 0
        print(" -- Epoch {}/{}".format(epoch, max_epoch))  # 1
        
        model = model.to(device)
        model.train()
        # model.train()
        # for batch_idx, ([data,
        #                  data_aug], target) in tqdm(enumerate(trainloader)):
        for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
        # for data, target in trainloader:
        #     print(data, target)

            images, labels = data.to(device), target.to(device)
            
            # output forward
            outputs = model(images)

            # set all gradients to zero
            optimizer.zero_grad()

            # fetch data
            # images, labels = data, target
            # if use_cuda:
            #     images = images.cuda()
            #     labels = labels.cuda()

            # normalization
            images = (images - images.mean()) / (images.std() + 1e-8)
            
            # calculate loss
            loss = criterion(outputs, labels)

            # loss2 = nce(inputs=outputs, labels=labels)

            # losses = sum(loss,loss2)

            # total_loss = sum(losses)
            # backward and optimize parameters
            # losses.backward()
            loss.backward()
            optimizer.step()

            # loss_list.append(loss.item())

            pred = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            # running_loss += losses.item()
            running_correct += torch.sum(pred == labels)
        
        # # record loss, accuracy
        loss = running_loss / len(trainset)
        # losses = running_loss / len(trainset)
        train_loss_list.append(loss)
        # loss_list.append(losses)
        
        acc_train = running_correct / len(trainset)
        acc_train_list.append(acc_train.item())

        # running_correct = (pred == labels).float()
        # acc_train = running_correct.sum() / running_correct.numel()
        # acc_train_list.append(acc_train.item())

            # pass

        model.eval()

        y_true = []
        y_pred = []
        running_loss = 0.0
        test_correct = 0
        # for _, ([data, data_aug], target) in enumerate(testloader):
        for _, (data, target) in enumerate(testloader):
            # fetch data
            images = data.to(device)
            labels = target.to(device)
            # predict = torch.argmax(net(data), dim=1)
            # images, labels = data, target
            # if use_cuda:
            #     images = images.cuda()
            #     labels = labels.cuda()

            # model forward
            outputs = model(images)

            # normalization
            images = (images - images.mean()) / (images.std() + 1e-8)

            loss = criterion(outputs, labels)
            # loss2 = nce(outputs, labels)

            # losses = [loss,loss2]

            # record the correct
            # predict = torch.argmax(outputs, dim=1)
            predict = torch.argmax(outputs, dim=1).cpu()#.data.numpy()
            y_pred.extend(predict)
            labels = target.cpu()#.data.numpy()
            y_true.extend(labels)

            # pred = torch.argmax(outputs, dim=1)
            running_loss += loss.item()
            # running_loss += losses.item()
            test_correct += torch.sum(predict == labels)

        loss = running_loss / len(testset)
        # losses = running_loss / len(testset)
        test_loss_list.append(loss)
        # loss_list.append(losses)

        acc_test = test_correct / len(testset)
        acc_test_list.append(acc_test.item())
            # y_pred = torch.augmax(outputs, dim=1)
            # y_true += torch.sum(y_pred == labels)
            # y_true.append(y_pred if y_pred == labels else '') ##
            # pass
        
        print("Loss {:.4f}, Train Accuracy {:.4f}%, Test Accuracy {:.4f}%".format(
        loss,
        acc_train * 100,
        acc_test * 100
        ))

        # acc = accuracy_score(y_true, y_pred) # torch.Tensor.tolist
        # print("=> Epoch:{} - Test Accuracy: {:.4f}".format(max_epoch, acc))
        
        # print("Loss {:.4f}, Test Accuracy {:.4f}%".format(
        #     loss,
        #     acc_test * 100
        # ))
        # print("Loss {:.4f}, Train Accuracy {:.4f}%".format(
        #     loss,
        #     acc_train * 100,
        # ))

    # x = np.arange(1, max_epoch+1)
    # y = loss_list
    # plt.title('Training Loss Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.plot(x, y)
    # # plt.show()
    # plt.savefig('p2-training_loss.svg')

    # x1 = np.arange(1, max_epoch+1)
    # y1 = acc_train_list
    # plt.title('Training Accuracy Curve')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.plot(x1, y1)
    # # plt.show()
    # plt.savefig('p2-training_accuracy.svg')
    # x = np.arange(1, max_epoch+1)
    # y1 = loss_list
    # y2 = acc_train_list
    # fig, ax1 = plt.subplots()
    # plt.title('Training Curves')
    # ax2 = ax1.twinx()
    # ax1.plot(x, y1, color = 'blue')
    # ax2.plot(x, y2, color = 'orange')
    # ax1.set_xlabel('Epoch', color = 'black')
    # ax1.set_ylabel('Loss', color = 'blue')
    
    # # secondary y-axis label
    # ax2.set_ylabel('Accuracy', color = 'orange')
    # fig.savefig('p2-training.jpg',
    #         format='jpeg',
    #         dpi=300)

    x = np.arange(1,max_epoch+1)
    y1 = train_loss_list
    y2 = test_loss_list
    plt.title('Training and Testing Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x,y1,color = 'green')
    plt.plot(x,y2, color = 'red')
    plt.legend(['Training Loss','Testing Loss'], loc = 'upper right')
    plt.savefig('p2-loss.jpg',
            format='jpeg',
            dpi=300)
    plt.close()

    x = np.arange(1,max_epoch+1)
    y1 = acc_train_list
    y2 = acc_test_list
    plt.title('Training and Testing Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(x,y1,color = 'green')
    plt.plot(x,y2, color = 'red')
    plt.legend(['Training Accuracy','Testing Accuracy'], loc = 'lower right')
    plt.savefig('p2-accuracy.jpg',
            format='jpeg',
            dpi=300)

# def test(model, testloader, max_epoch):
#     model.eval()

#     y_true = []
#     y_pred = []
#     running_loss = 0.0
#     test_correct = 0
#     loss_list = []
#     acc_test_list = []
#     # for _, ([data, data_aug], target) in enumerate(testloader):
#     for _, (data, target) in enumerate(testloader):
#         # fetch data
#         images = data.to(device)
#         labels = target.to(device)
#         # predict = torch.argmax(net(data), dim=1)
#         # images, labels = data, target
#         # if use_cuda:
#         #     images = images.cuda()
#         #     labels = labels.cuda()

#         # model forward
#         outputs = model(images)

#         # normalization
#         images = (images - images.mean()) / (images.std() + 1e-8)

#         loss = criterion(outputs, labels)
#         # loss2 = nce(outputs, labels)

#         # losses = [loss,loss2]

#         # record the correct
#         # predict = torch.argmax(outputs, dim=1)
#         predict = torch.argmax(outputs, dim=1).cpu()#.data.numpy()
#         y_pred.extend(predict)
#         labels = target.cpu()#.data.numpy()
#         y_true.extend(labels)

#         # pred = torch.argmax(outputs, dim=1)
#         running_loss += loss.item()
#         # running_loss += losses.item()
#         test_correct += torch.sum(predict == labels)

#     loss = running_loss / len(testset)
#     # losses = running_loss / len(testset)
#     loss_list.append(loss)
#     # loss_list.append(losses)

#     acc_test = test_correct / len(testset)
#     acc_test_list.append(acc_test.item())
#         # y_pred = torch.augmax(outputs, dim=1)
#         # y_true += torch.sum(y_pred == labels)
#         # y_true.append(y_pred if y_pred == labels else '') ##
#         # pass

#     acc = accuracy_score(y_true, y_pred) # torch.Tensor.tolist
#     print("=> Epoch:{} - Test Accuracy: {:.4f}".format(max_epoch, acc))
    
#     print("Loss {:.4f}, Test Accuracy {:.4f}%".format(
#         loss,
#         acc_test * 100
#     ))
#     # x = np.arange(1, max_epoch+1)
#     # y = loss_list
#     # plt.title('Testing Loss Curve')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # plt.plot(x, y)
#     # # plt.show()
#     # plt.savefig('p2-testing_loss.svg')

#     # x1 = np.arange(1, max_epoch+1)
#     # y1 = acc_test_list
#     # plt.title('Testing Accuracy Curve')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Accuracy')
#     # plt.plot(x1, y1)
#     # # plt.show()
#     # plt.savefig('p2-testing_accuracy.svg')
#     # return acc

if __name__ == "__main__":
    # print("Loss {:.4f}, Train Accuracy {:.4f}%, Test Accuracy {:.4f}%".format(
    #     # loss,
    #     *train(model=net, trainloader=trainloader, use_cuda=use_cuda) * 100,
    #     test(model=net, testloader=testloader, epoch=max_epoch, use_cuda=use_cuda) * 100
    # ))
    train(model=net, trainloader=trainloader, testloader=testloader, max_epoch=max_epoch, optimizer=optimizer)
    # test(model=net, testloader=testloader, max_epoch=max_epoch)

    # x = np.arange(1,max_epoch+1)
    # y1 = acc_train_list
    # y2 = acc_test_list
    # plt.title("Training and testing accuracy") 
    # plt.xlabel("Epoch") 
    # plt.ylabel("Accuracy") 
    # plt.plot(x,y1)
    # plt.plot(x,y2)
    # plt.legend(['Training accuracy', 'Testing accuracy'])
    # plt.savefig("p2-accuracy.jpg")
    
    # x = np.arange(1,max_epoch+1)
    # y1 = train_loss_list
    # y2 = test_loss_list
    # plt.title("Training and testing accuracy") 
    # plt.xlabel("Epoch") 
    # plt.ylabel("LOss") 
    # plt.plot(x,y1)
    # plt.plot(x,y2)
    # plt.legend(['Training Loss', 'Testing Loss'])
    # plt.savefig("p2-loss.jpg")