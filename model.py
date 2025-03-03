import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def conv(in_channels, out_channels, kernel_size, stride=1):
    '''
        Convolutional layer with batch normalization and ReLU activation
    '''
    padding = (kernel_size - 1) // 2 # Ensure the same output size as input size
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    

class MobileNetBlock(nn.Module):
    '''
        Adapt from MobileNet
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        self.depthwise = conv(in_channels, in_channels, 3, stride) # Capturing spatial information
        self.pointwise = conv(in_channels, out_channels, 1, stride) # Increasing the depth
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
    
class Encoder(nn.Module):
    '''
        Input: N * 4 * H * W (4 = rgb + depth)
        Output: N * 512 * H/4 * W/4
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_channels = config.config["input_channels"]
        self.conv1 = MobileNetBlock(self.in_channels, 32)
        self.conv2 = MobileNetBlock(32, 64)
        self.conv3 = MobileNetBlock(64, 128)
        # The original paper suggests to use 5 layers, but we only use 128->256->512
        self.conv4 = MobileNetBlock(128, 256)
        self.conv5 = MobileNetBlock(256, 512)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # torch.Size([4, 32, 520, 240]) torch.Size([4, 64, 520, 240]) torch.Size([4, 128, 520, 240]) torch.Size([4, 256, 520, 240]) torch.Size([4, 512, 520, 240])
        return x1, x2, x3, x4, x5
     
    
class Decoder(nn.Module):
    '''
        Input: 5 layers of features from Encoder
        Output: N * 1 (depth) * H * W
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        self.out_channels = config.config["output_channels"]
        
        self.conv1 = MobileNetBlock(512, 256)
        self.conv2 = MobileNetBlock(256, 128)
        self.conv3 = MobileNetBlock(128, 64)
        self.conv4 = MobileNetBlock(64, 32)
        self.conv5 = MobileNetBlock(32, self.out_channels) # Output depth only
        
    def forward(self, x1, x2, x3, x4, x5):
        x = self.conv1(x5)
        x = self.conv2(x) + x3
        x = self.conv3(x) + x2
        x = self.conv4(x) + x1
        return self.conv5(x)
        
        
class DepthModel(nn.Module):
    '''
        DepthModel: Encoder + Decoder (with bottleneck)
    '''
    def __init__(self):
        super(DepthModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bottle_neck = nn.Conv2d(512, 512, kernel_size=1)
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.bottle_neck(x5)
        return self.decoder(x1, x2, x3, x4, x5)