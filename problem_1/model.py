import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        # t4 = out
        # out = F.relu(F.max_pool3d(self.encoder5(out),2,2))
        
        # t2 = out
        # out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

# Another design 1
# import torch
# from torch import nn

# class EncoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout=False):
#         super(EncoderBlock, self).__init__()
#         self.encode = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#         if dropout:
#             self.encode.add_module('dropout', nn.Dropout())
#         self.encode.add_module('maxpool', nn.MaxPool2d(2, stride=2))

#     def forward(self, x):
#         return self.encode(x)

# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.decode = nn.Sequential(
#             nn.Conv2d(in_channels, middle_channels, 3, padding=1),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(middle_channels, middle_channels, 3, padding=1),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(middle_channels, out_channels, 2, stride=2),
#         )
       
#     def forward(self, x):
#         return self.decode(x)

# class UNet(nn.Module):
#     def __init__(self, num_classes):
#         super(UNet, self).__init__()
#         self.encoder1 = EncoderBlock(3, 64)
#         self.encoder2 = EncoderBlock(64, 128)
#         self.encoder3 = EncoderBlock(128, 256)
#         self.encoder4 = EncoderBlock(256, 512)

#         self.center = DecoderBlock(512, 1024, 512)

#         self.decoder4 = DecoderBlock(1024, 512, 256)
#         self.decoder3 = DecoderBlock(512, 256, 128)
#         self.decoder2 = DecoderBlock(256, 128, 64)
#         self.decoder1 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, num_classes, 1),
#         )

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(enc1)
#         enc3 = self.encoder3(enc2)
#         enc4 = self.encoder4(enc3)
#         x = self.center(enc4)
#         x = self.decoder4(torch.cat((enc4, x), dim=1))
#         x = self.decoder3(torch.cat((enc3, x), dim=1))
#         x = self.decoder2(torch.cat((enc2, x), dim=1))
#         x = self.decoder1(torch.cat((enc1, x), dim=1))
#         return x

# Another design 2
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # def conv_block(in_chan, out_chan, stride=1):
# #     return nn.Sequential(
# #         nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
# #         nn.BatchNorm3d(out_chan),
# #         nn.ReLU(inplace=True)
# #     )


# # def conv_stage(in_chan, out_chan):
# #     return nn.Sequential(
# #         conv_block(in_chan, out_chan),
# #         conv_block(out_chan, out_chan),
# #     )


# # class UNet(nn.Module):

# #     def __init__(self):
# #         super().__init__()

# #         self.enc1 = conv_stage(1, 16)
# #         self.enc2 = conv_stage(16, 32)
# #         self.enc3 = conv_stage(32, 64)
# #         self.enc4 = conv_stage(64, 128)
# #         self.enc5 = conv_stage(128, 128)
# #         self.pool = nn.MaxPool3d(2, 2)

# #         self.dec4 = conv_stage(256, 64)
# #         self.dec3 = conv_stage(128, 32)
# #         self.dec2 = conv_stage(64, 16)
# #         self.dec1 = conv_stage(32, 16)
# #         self.conv_out = nn.Conv3d(16, 1, 1)

# #     def forward(self, x):
# #         enc1 = self.enc1(x)
# #         enc2 = self.enc2(self.pool(enc1))
# #         enc3 = self.enc3(self.pool(enc2))
# #         enc4 = self.enc4(self.pool(enc3))
# #         enc5 = self.enc5(self.pool(enc4))

# #         dec4 = self.dec4(torch.cat((enc4, F.upsample(enc5, enc4.size()[2:], mode='trilinear')), 1))
# #         dec3 = self.dec3(torch.cat((enc3, F.upsample(dec4, enc3.size()[2:], mode='trilinear')), 1))
# #         dec2 = self.dec2(torch.cat((enc2, F.upsample(dec3, enc2.size()[2:], mode='trilinear')), 1))
# #         dec1 = self.dec1(torch.cat((enc1, F.upsample(dec2, enc1.size()[2:], mode='trilinear')), 1))
# #         out = self.conv_out(dec1)
# #         out = F.sigmoid(out)
# #         return out