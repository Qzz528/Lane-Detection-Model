# -*- coding: utf-8 -*-
"""
FCN without Crop (by padding in the Conv, ConvTranspose)
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16

#将卷积层，激活函数集成为一个模块
#预设stride=1，且kernel_size为2倍padding+1 可保证数据通过该模块处理后宽高尺寸不变
class conv_relu(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,h,w)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                              stride = 1, padding = 1,bias = False)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class FCN(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)  
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 pretrained = False, #模型features部分是否使用预训练的vgg16模型
                 rate = 8, #FCN的结构，可以为8,16,32
                 ):
        super().__init__()
        self.rate = rate
        assert rate in [8,16,32]
        
        '模型结构'
        ###feature #对输入数据尺寸进行压缩至输入尺寸的1/32
        if pretrained:
            backbone = vgg16(pretrained).features
            self.features3 = backbone[:17] #1->1/8
            self.features4 = backbone[17:24] #1/8->1/16
            self.features5 = backbone[24:] #1/16->1/32
            
        else:
            self.features3 = nn.Sequential(
                #1->1/2
                conv_relu(3, 64),
                conv_relu(64, 64),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
    
                #1/2->1/4
                conv_relu(64, 128),
                conv_relu(128, 128),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
    
                #1/4->1/8
                conv_relu(128, 256),
                conv_relu(256, 256),
                conv_relu(256, 256),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
            )
            self.features4 = nn.Sequential(
                #1/8->1/16
                conv_relu(256, 512),
                conv_relu(512, 512),
                conv_relu(512, 512),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
            )
            self.features5 = nn.Sequential(
                #1/16->1/32
                conv_relu(512, 512),
                conv_relu(512, 512),
                conv_relu(512, 512),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
            )
        
               
        ###classifier #将尺寸压缩后数据的通道数转为要分类的类别数n_cls
        self.classifier5 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, n_cls, kernel_size=1),
        )
        if rate <= 16:
            self.classifier4 = nn.Conv2d(512, n_cls, kernel_size=1)
            if rate <= 8:
                self.classifier3 = nn.Conv2d(256, n_cls, kernel_size=1)
        
        
        ###deconv #将通道为c_cls数据的尺寸恢复为原图尺寸
        if rate == 32:
            #1/32->1
            self.deconv5 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size=64, 
                                              stride=32, padding=16, bias=False)
        elif rate == 16:
            #1/32->1/16
            self.deconv5 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size=4, 
                                              stride=2, padding=1, bias=False)
            #1/16->1
            self.deconv4 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size = 32, 
                                              stride=16, padding = 8, bias=False)

        elif rate == 8:
            #1/32->1/16
            self.deconv5 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size=4, 
                                              stride=2, padding=1, bias=False)
            #1/16->1/8
            self.deconv4 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size = 4, 
                                              stride=2, padding = 1, bias=False)
            #1/8->1
            self.deconv3 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size=16, 
                                              stride=8, padding=4, bias=False)
        
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, x):#(bs,3,h,w)
        
        feat3 = self.features3(x)  #(bs,256,h/8,w/8)
        feat4 = self.features4(feat3)  #(bs,512,h/16,w/16)
        feat5 = self.features5(feat4)  #(bs,512,h/32,w/32)
    

        score5 = self.classifier5(feat5) #(bs,n_cls,h/32,w/32)
                
        if self.rate==32:
            return self.deconv5(score5) #(bs,n_cls,h,w)
        
        
        up5 = self.deconv5(score5) #rate16&8(bs,n_cls,h/16,w/16)
        score4 = self.classifier4(feat4) #(bs,n_cls,h/16,w/16)
        score4 += up5
        
        
        if self.rate==16:
            return self.deconv4(score4) #(bs,n_cls,h,w)
        
        up4 = self.deconv4(score4)#rate8(bs,n_cls,h/8,w/8)
        score3 = self.classifier3(feat3) #(bs,n_cls,h/8,w/8)
        score3 += up4
        out = self.deconv3(score3) #rate8(bs,n_cls,h,w)
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况
        
        return out #(bs,n_cls,h,w) 

    def loss(self, pred, targ):
        #pred:(bs,n_cls,h,w) || targ:(bs,h,w) 取值0~n_cls-1
        return self.celoss(pred,targ)
    
if __name__ == '__main__':
    inputs = torch.zeros(1,3,224,224)
    n_cls = 11
    
    for pretrained in [True,False]:
        for rate in [32,16,8]:
            model = FCN(n_cls,pretrained,rate)
            outputs = model(inputs)
            print(f'classes: {n_cls} || pretrained: {pretrained} || structure: {rate}')
            print(f'inputshape:{inputs.shape} || outputshape:{outputs.shape}\n')
                      

#自写的仿结构的低参数量模型
class FCN_small(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)  
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 ):
        super().__init__()
        
        '模型结构'
        self.features3 = nn.Sequential(
            #1->1/2
            conv_relu(3, 8),
            conv_relu(8, 8),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  

            #1/2->1/4
            conv_relu(8, 16),
            conv_relu(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  

            #1/4->1/8
            conv_relu(16, 32),
            conv_relu(32, 32),
            conv_relu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
        )
        self.features4 = nn.Sequential(
            #1/8->1/16
            conv_relu(32, 64),
            conv_relu(64, 64),
            conv_relu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
        )
        self.features5 = nn.Sequential(
            #1/16->1/32
            conv_relu(64, 64),
            conv_relu(64, 64),
            conv_relu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  
        )
    
               
        ###classifier #将尺寸压缩后数据的通道数转为要分类的类别数n_cls
        self.classifier5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(128, n_cls, kernel_size=1),
        )
        
        self.classifier4 = nn.Conv2d(64, n_cls, kernel_size=1)

        self.classifier3 = nn.Conv2d(32, n_cls, kernel_size=1)
        
        
        ###deconv #将通道为c_cls数据的尺寸恢复为原图尺寸
        #1/32->1/16
        self.deconv5 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size=4, 
                                          stride=2, padding=1, bias=False)
        #1/16->1/8
        self.deconv4 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size = 4, 
                                          stride=2, padding = 1, bias=False)
        #1/8->1
        self.deconv3 = nn.ConvTranspose2d(n_cls, n_cls, kernel_size=16, 
                                          stride=8, padding=4, bias=False)
        
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, x):#(bs,3,h,w)
        
        feat3 = self.features3(x)  #(bs,256,h/8,w/8)
        feat4 = self.features4(feat3)  #(bs,512,h/16,w/16)
        feat5 = self.features5(feat4)  #(bs,512,h/32,w/32)
    

        score5 = self.classifier5(feat5) #(bs,n_cls,h/32,w/32)
       
        up5 = self.deconv5(score5) #(bs,n_cls,h/16,w/16)
        score4 = self.classifier4(feat4) #(bs,n_cls,h/16,w/16)
        score4 += up5
        
        up4 = self.deconv4(score4)#rate8(bs,n_cls,h/8,w/8)
        score3 = self.classifier3(feat3) #(bs,n_cls,h/8,w/8)
        score3 += up4
        
        out = self.deconv3(score3) #rate8(bs,n_cls,h,w)
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况
        
        return out #(bs,n_cls,h,w) 

    def loss(self, pred, targ):
        #pred:(bs,n_cls,h,w) || targ:(bs,h,w) 取值0~n_cls-1
        return self.celoss(pred,targ)