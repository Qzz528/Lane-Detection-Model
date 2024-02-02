# -*- coding: utf-8 -*-
"""
UNet without crop (by padding in the Conv)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


#将两轮的卷积、批标准化、激活函数集成为一个模块
#预设stride=1，且kernel_size为2倍padding+1 可保证数据通过该模块处理后宽高尺寸不变
class conv_block(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,h,w)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

#将上采样、卷积、批标准化、激活函数集成为一个模块
#预设stride=1，且kernel_size为2倍padding+1 可保证数据通过该模块处理后宽高尺寸变为原来2倍
class upconv_block(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,2h,2w)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)
    #https://arxiv.org/abs/1505.04597    
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 ):
        super().__init__()
        
        '模型结构'
        ###features(encoder) #对输入数据尺寸进行压缩至输入尺寸的1/16
        #1->1/2
        self.encoder1 = conv_block(3, 64) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #1/2->1/4
        self.encoder2 = conv_block(64, 128) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #1/4->1/8
        self.encoder3 = conv_block(128, 256) 
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #1/8->1/16
        self.encoder4 = conv_block(256, 512) 
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        #1/16->1/16
        self.feat_conv = conv_block(512, 1024)

        ###unpool(decoder)  #与features(encoder)对称
        #将尺寸压缩后数据的尺寸恢复为原图尺寸，并把通道数转为要分类的类别数n_cls
        #1/16->1/8
        self.up4 = upconv_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)

        #1/8->1/4
        self.up3 = upconv_block(512, 256) 
        self.decoder3 = conv_block(512, 256) 

        #1/4->1/2
        self.up2 = upconv_block(256, 128) 
        self.decoder2 = conv_block(256, 128) 

        #1/2->1
        self.up1 = upconv_block(128, 64) 
        self.decoder1 = conv_block(128, 64) 

        self.final_conv = nn.Conv2d(64, n_cls, kernel_size=1, stride=1, padding=0)
        
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, x):#(bs,3,h,w)  
        
        #encoding
        e1 = self.encoder1(x) #e1:(bs,64,h,w)  

        e2 = self.maxpool1(e1) #(bs,64,h/2,w/2)
        e2 = self.encoder2(e2) #e2:(bs,128,h/2,w/2)  

        e3 = self.maxpool2(e2) #(bs,128,h/4,w/4)
        e3 = self.encoder3(e3) #e3:(bs,256,h/4,w/4)  

        e4 = self.maxpool3(e3) #(bs,256,h/8,w/8)  
        e4 = self.encoder4(e4) #e4:(bs,512,h/8,w/8)  

        feat = self.maxpool4(e4) #(bs,512,h/16,w/16)  
        feat = self.feat_conv(feat) #feat:(bs,1024,h/16,w/16)  
        
        #decoding
        d4 = self.up4(feat) #d4:(bs,512,h/8,w/8)
        d4 = torch.cat((e4, d4), dim=1) #(bs,1024,h/8,w/8)  
        d4 = self.decoder4(d4) #d4:(bs,512,h/8,w/8)  

        d3 = self.up3(d4) #d3:(bs,256,h/4,w/4)
        d3 = torch.cat((e3, d3), dim=1) #(bs,512,h/4,w/4)  
        d3 = self.decoder3(d3) #d3:(bs,256,h/4,w/4)  

        d2 = self.up2(d3) #d2:(bs,128,h/2,w/2)
        d2 = torch.cat((e2, d2), dim=1) #(bs,256,h/2,w/2)  
        d2 = self.decoder2(d2) #d2:(bs,128,h/2,w/2)  

        d1 = self.up1(d2) #d1:(bs,64,h,w)
        d1 = torch.cat((e1, d1), dim=1) #(bs,128,h,w)  
        d1 = self.decoder1(d1) #d1:(bs,64,h,w)  

        out = self.final_conv(d1) #out:(bs,n_cls,h,w)
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况

        return out #(bs,n_cls,h,w)
    
    def loss(self, pred, targ):
        #pred:(bs,n_cls,h,w) || targ:(bs,h,w) 取值0~n_cls-1
        return self.celoss(pred,targ)


if __name__ == '__main__':
    inputs = torch.zeros(1,3,224,224)
    n_cls = 11
   
    model = UNet(n_cls)
    outputs = model(inputs)
    print(f'classes: {n_cls}')
    print(f'inputshape:{inputs.shape} || outputshape:{outputs.shape}\n')
    
#自写的仿结构的低参数量模型     
class UNet_small(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 ):
        super().__init__()
        
        '模型结构'
        ###features(encoder) #对输入数据尺寸进行压缩至输入尺寸的1/16
        #1->1/2
        self.encoder1 = conv_block(3, 8) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #1/2->1/4
        self.encoder2 = conv_block(8, 16) 
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #1/4->1/8
        self.encoder3 = conv_block(16, 32) 
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #1/8->1/16
        self.encoder4 = conv_block(32, 64) 
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        #1/16->1/16
        self.feat_conv = conv_block(64, 128)

        ###unpool(decoder)  #与features(encoder)对称
        #将尺寸压缩后数据的尺寸恢复为原图尺寸，并把通道数转为要分类的类别数n_cls
        #1/16->1/8
        self.up4 = upconv_block(128, 64)
        self.decoder4 = conv_block(128, 64)

        #1/8->1/4
        self.up3 = upconv_block(64, 32) 
        self.decoder3 = conv_block(64, 32) 

        #1/4->1/2
        self.up2 = upconv_block(32, 16) 
        self.decoder2 = conv_block(32, 16) 

        #1/2->1
        self.up1 = upconv_block(16, 8) 
        self.decoder1 = conv_block(16, 8) 

        self.final_conv = nn.Conv2d(8, n_cls, kernel_size=1, stride=1, padding=0)
        
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, x):#(bs,3,h,w)  
        
        #encoding
        e1 = self.encoder1(x) #e1:(bs,64,h,w)  

        e2 = self.maxpool1(e1) #(bs,64,h/2,w/2)
        e2 = self.encoder2(e2) #e2:(bs,128,h/2,w/2)  

        e3 = self.maxpool2(e2) #(bs,128,h/4,w/4)
        e3 = self.encoder3(e3) #e3:(bs,256,h/4,w/4)  

        e4 = self.maxpool3(e3) #(bs,256,h/8,w/8)  
        e4 = self.encoder4(e4) #e4:(bs,512,h/8,w/8)  

        feat = self.maxpool4(e4) #(bs,512,h/16,w/16)  
        feat = self.feat_conv(feat) #feat:(bs,1024,h/16,w/16)  
        
        #decoding
        d4 = self.up4(feat) #d4:(bs,512,h/8,w/8)
        d4 = torch.cat((e4, d4), dim=1) #(bs,1024,h/8,w/8)  
        d4 = self.decoder4(d4) #d4:(bs,512,h/8,w/8)  

        d3 = self.up3(d4) #d3:(bs,256,h/4,w/4)
        d3 = torch.cat((e3, d3), dim=1) #(bs,512,h/4,w/4)  
        d3 = self.decoder3(d3) #d3:(bs,256,h/4,w/4)  

        d2 = self.up2(d3) #d2:(bs,128,h/2,w/2)
        d2 = torch.cat((e2, d2), dim=1) #(bs,256,h/2,w/2)  
        d2 = self.decoder2(d2) #d2:(bs,128,h/2,w/2)  

        d1 = self.up1(d2) #d1:(bs,64,h,w)
        d1 = torch.cat((e1, d1), dim=1) #(bs,128,h,w)  
        d1 = self.decoder1(d1) #d1:(bs,64,h,w)  

        out = self.final_conv(d1) #out:(bs,n_cls,h,w)
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况

        return out #(bs,n_cls,h,w)
    
    def loss(self, pred, targ):
        #pred:(bs,n_cls,h,w) || targ:(bs,h,w) 取值0~n_cls-1
        return self.celoss(pred,targ)