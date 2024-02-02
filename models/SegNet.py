# -*- coding: utf-8 -*-
"""
SegNet
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn


#将卷积层，批归一化，激活函数集成为一个模块
#预设stride=1，且kernel_size为2倍padding+1 可保证数据通过该模块处理后宽高尺寸不变
class conv_bn_relu(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,h,w)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                                    stride = 1, padding = 1,bias = False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SegNet(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)  
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 pretrained = False, #模型features部分是否使用预训练的vgg16bn模型
                 ):
        super().__init__()    
        
        '模型结构'
        ###features(encoder) #对输入数据尺寸进行压缩至输入尺寸的1/32
        if pretrained:
            #从vgg中提取features的层，分配给encoder
            backbone = vgg16_bn(pretrained).features
            #1->1/2
            self.encoder1 = backbone[0:6] 
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            #1/2->1/4
            self.encoder2 = backbone[7:13] 
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            
            #1/4->1/8
            self.encoder3 = backbone[14:23] 
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            
            #1/8->1/16
            self.encoder4 = backbone[24:33]
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
           
            #1/16->1/32
            self.encoder5 = backbone[34:43]
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        else:
            #1->1/2
            self.encoder1 = nn.Sequential(
                conv_bn_relu(3, 64),
                conv_bn_relu(64, 64),
                )
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            #1/2->1/4
            self.encoder2 = nn.Sequential(
                conv_bn_relu(64, 128),
                conv_bn_relu(128, 128),
                )
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            
            #1/4->1/8
            self.encoder3 = nn.Sequential(
                conv_bn_relu(128, 256),
                conv_bn_relu(256, 256),
                conv_bn_relu(256, 256),
                )
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            #1/8->1/16
            self.encoder4 = nn.Sequential(
                conv_bn_relu(256, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                )
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            #1/16->1/32
            self.encoder5 = nn.Sequential(
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                conv_bn_relu(512, 512),
                )
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        
        ###unpool(decoder)  #与features(encoder)对称
        #将尺寸压缩后数据的尺寸恢复为原图尺寸，并把通道数转为要分类的类别数n_cls
        #1/32->1/16
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            )
 
        #1/16->1/8
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 256),
            )
    
        #1/8->1/4
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 128),
            )          
        
        #1/4->1/2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 64),
            )
         
        #1/2->1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            conv_bn_relu(64, 64),
            nn.Conv2d(64, n_cls, kernel_size=3, padding=1),#最后一个卷积不带bn和relu
            )
     
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, x):#(bs,3,h,w)       
          
        #encoding
        #id是池化时最大值所在的位置，反池化时要借助其将最大值填充回对应位置
        x = self.encoder1(x) #(bs,64,h,w)
        x,id1 = self.pool1(x) #(bs,64,h/2,w/2)
        
        x = self.encoder2(x) #(bs,128,h/2,w/2)
        x,id2 = self.pool2(x) #(bs,128,h/4,w/4)

        x = self.encoder3(x) #(bs,256,h/4,w/4)
        x,id3 = self.pool3(x) #(bs,256,h/8,w/8)  

        x = self.encoder4(x) #(bs,512,h/8,w/8)
        x,id4 = self.pool4(x) #(bs,512,h/16,w/16)

        x = self.encoder5(x) #(bs,512,h/16,w/16)
        x,id5 = self.pool5(x) #(bs,512,h/32,w/32)             
    
        #decoding
        x = self.unpool5(x,id5) #(bs,512,h/16,w/16)
        x = self.decoder5(x) #(bs,512,h/16,w/16)
        
        x = self.unpool4(x,id4) #(bs,512,h/8,w/8)
        x = self.decoder4(x) #(bs,256,h/8,w/8)        
        
        x = self.unpool3(x,id3) #(bs,256,h/4,w/4)
        x = self.decoder3(x) #(bs,128,h/4,w/4)         
        
        x = self.unpool2(x,id2) #(bs,128,h/2,w/2)
        x = self.decoder2(x) #(bs,64,h/2,w/2)           
        
        x = self.unpool1(x,id1) #(bs,64,h,w)
        x = self.decoder1(x) #(bs,n_cls,h,w)            
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况
        
        return x #(bs,n_cls,h,w) 
    
    def loss(self, pred, targ):
        #pred:(bs,n_cls,h,w) || targ:(bs,h,w) 取值0~n_cls-1
        return self.celoss(pred,targ)


if __name__ == '__main__':
    inputs = torch.zeros(1,3,224,224)
    n_cls = 11
    
    for pretrained in [True,False]:
        model = SegNet(n_cls,pretrained)
        outputs = model(inputs)
        print(f'classes: {n_cls} || pretrained: {pretrained}')
        print(f'inputshape:{inputs.shape} || outputshape:{outputs.shape}\n')
        
#自写的仿结构的低参数量模型        
class SegNet_small(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)  
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 ):
        super().__init__()    
        
        '模型结构'
        ###features(encoder) #对输入数据尺寸进行压缩至输入尺寸的1/32
        #1->1/2
        self.encoder1 = nn.Sequential(
            conv_bn_relu(3, 8),
            conv_bn_relu(8, 8),
            )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        #1/2->1/4
        self.encoder2 = nn.Sequential(
            conv_bn_relu(8, 16),
            conv_bn_relu(16, 16),
            )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        #1/4->1/8
        self.encoder3 = nn.Sequential(
            conv_bn_relu(16, 32),
            conv_bn_relu(32, 32),
            conv_bn_relu(32, 32),
            )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        #1/8->1/16
        self.encoder4 = nn.Sequential(
            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        #1/16->1/32
        self.encoder5 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    
        ###unpool(decoder)  #与features(encoder)对称
        #将尺寸压缩后数据的尺寸恢复为原图尺寸，并把通道数转为要分类的类别数n_cls
        #1/32->1/16
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder5 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            )
 
        #1/16->1/8
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 32),
            )
    
        #1/8->1/4
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            conv_bn_relu(32, 32),
            conv_bn_relu(32, 32),
            conv_bn_relu(32, 16),
            )          
        
        #1/4->1/2
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            conv_bn_relu(16, 16),
            conv_bn_relu(16, 8),
            )
         
        #1/2->1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            conv_bn_relu(8, 8),
            nn.Conv2d(8, n_cls, kernel_size=3, padding=1),#最后一个卷积不带bn和relu
            )
     
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()
        
    def forward(self, x):#(bs,3,h,w)       
          
        #encoding
        #id是池化时最大值所在的位置，反池化时要借助其将最大值填充回对应位置
        x = self.encoder1(x) #(bs,64,h,w)
        x,id1 = self.pool1(x) #(bs,64,h/2,w/2)
        
        x = self.encoder2(x) #(bs,128,h/2,w/2)
        x,id2 = self.pool2(x) #(bs,128,h/4,w/4)

        x = self.encoder3(x) #(bs,256,h/4,w/4)
        x,id3 = self.pool3(x) #(bs,256,h/8,w/8)  

        x = self.encoder4(x) #(bs,512,h/8,w/8)
        x,id4 = self.pool4(x) #(bs,512,h/16,w/16)

        x = self.encoder5(x) #(bs,512,h/16,w/16)
        x,id5 = self.pool5(x) #(bs,512,h/32,w/32)             
    
        #decoding
        x = self.unpool5(x,id5) #(bs,512,h/16,w/16)
        x = self.decoder5(x) #(bs,512,h/16,w/16)
        
        x = self.unpool4(x,id4) #(bs,512,h/8,w/8)
        x = self.decoder4(x) #(bs,256,h/8,w/8)        
        
        x = self.unpool3(x,id3) #(bs,256,h/4,w/4)
        x = self.decoder3(x) #(bs,128,h/4,w/4)         
        
        x = self.unpool2(x,id2) #(bs,128,h/2,w/2)
        x = self.decoder2(x) #(bs,64,h/2,w/2)           
        
        x = self.unpool1(x,id1) #(bs,64,h,w)
        x = self.decoder1(x) #(bs,n_cls,h,w)            
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况
        
        return x #(bs,n_cls,h,w) 
    
    def loss(self, pred, targ):
        #pred:(bs,n_cls,h,w) || targ:(bs,h,w) 取值0~n_cls-1
        return self.celoss(pred,targ)