import torch
import torch.nn as nn
import torchvision
import numpy as np

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,
                  out_channels=places,
                  kernel_size=7,
                  stride=stride,
                  padding=3,
                  bias=False), nn.BatchNorm2d(places), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


class Bottleneck(nn.Module):

    def __init__(self,
                 in_places,
                 places,
                 stride=1,
                 downsampling=False,
                 expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,
                      out_channels=places,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places,
                      out_channels=places,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places,
                      out_channels=places * self.expansion,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places,
                          out_channels=places * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(places * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes=3, places=64)

        self.layer1 = self.make_layer(in_places=64,
                                      places=64,
                                      block=blocks[0],
                                      stride=1)
        self.layer2 = self.make_layer(in_places=256,
                                      places=128,
                                      block=blocks[1],
                                      stride=2)
        self.layer3 = self.make_layer(in_places=512,
                                      places=256,
                                      block=blocks[2],
                                      stride=2)
        self.layer4 = self.make_layer(in_places=1024,
                                      places=512,
                                      block=blocks[3],
                                      stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet50():
    return ResNet([3, 4, 6, 3])


def ResNet101():
    return ResNet([3, 4, 23, 3])


def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__ == '__main__':
    #model = torchvision.models.resnet50()
    model = ResNet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

# Another design 1
# import torch
# from torch import nn

# class ConvBlock(nn.Module):

#     def __init__(self, in_channel, f, filters, s):
#         super(ConvBlock, self).__init__()
#         F1, F2, F3 = filters
#         self.stage = nn.Sequential(
#             nn.Conv2d(in_channel, F1, 1, stride=s, padding=0, bias=False),
#             nn.BatchNorm2d(F1),
#             nn.ReLU(True),
#             nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
#             nn.BatchNorm2d(F2),
#             nn.ReLU(True),
#             nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(F3),
#         )
#         self.shortcut_1 = nn.Conv2d(in_channel,
#                                     F3,
#                                     1,
#                                     stride=s,
#                                     padding=0,
#                                     bias=False)
#         self.batch_1 = nn.BatchNorm2d(F3)
#         self.relu_1 = nn.ReLU(True)

#     def forward(self, X):
#         X_shortcut = self.shortcut_1(X)
#         X_shortcut = self.batch_1(X_shortcut)
#         X = self.stage(X)
#         X = X + X_shortcut
#         X = self.relu_1(X)
#         return X

# class IndentityBlock(nn.Module):

#     def __init__(self, in_channel, f, filters):
#         super(IndentityBlock, self).__init__()
#         F1, F2, F3 = filters
#         self.stage = nn.Sequential(
#             nn.Conv2d(in_channel, F1, 1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(F1),
#             nn.ReLU(True),
#             nn.Conv2d(F1, F2, f, stride=1, padding=True, bias=False),
#             nn.BatchNorm2d(F2),
#             nn.ReLU(True),
#             nn.Conv2d(F2, F3, 1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(F3),
#         )
#         self.relu_1 = nn.ReLU(True)

#     def forward(self, X):
#         X_shortcut = X
#         X = self.stage(X)
#         X = X + X_shortcut
#         X = self.relu_1(X)
#         return X

# class ResModel(nn.Module):

#     def __init__(self, n_class):
#         super(ResModel, self).__init__()
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             nn.MaxPool2d(3, 2, padding=1),
#         )
#         self.stage2 = nn.Sequential(
#             ConvBlock(64, f=3, filters=[64, 64, 256], s=1),
#             IndentityBlock(256, 3, [64, 64, 256]),
#             IndentityBlock(256, 3, [64, 64, 256]),
#         )
#         self.stage3 = nn.Sequential(
#             ConvBlock(256, f=3, filters=[128, 128, 512], s=2),
#             IndentityBlock(512, 3, [128, 128, 512]),
#             IndentityBlock(512, 3, [128, 128, 512]),
#             IndentityBlock(512, 3, [128, 128, 512]),
#         )
#         self.stage4 = nn.Sequential(
#             ConvBlock(512, f=3, filters=[256, 256, 1024], s=2),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#             IndentityBlock(1024, 3, [256, 256, 1024]),
#         )
#         self.stage5 = nn.Sequential(
#             ConvBlock(1024, f=3, filters=[512, 512, 2048], s=2),
#             IndentityBlock(2048, 3, [512, 512, 2048]),
#             IndentityBlock(2048, 3, [512, 512, 2048]),
#         )
#         self.pool = nn.AvgPool2d(2, 2, padding=1)
#         self.fc = nn.Sequential(nn.Linear(8192, n_class))

#     def forward(self, X):
#         out = self.stage1(X)
#         out = self.stage2(out)
#         out = self.stage3(out)
#         out = self.stage4(out)
#         out = self.stage5(out)
#         out = self.pool(out)
#         out = out.view(out.size(0), 8192)
#         out = self.fc(out)
#         return out
