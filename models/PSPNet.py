# -*- coding: utf-8 -*-
"""
PSPNet without dilation in conv
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d

#将卷积层，激活函数集成为一个模块
class BasicBlock(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,h/2,w/2) if shortcut_conv else (bs,out_channels,h,w)
    def __init__(self, in_channels, out_channels, shortcut_conv = True):
        super().__init__()
        self.conv = shortcut_conv
        
        
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size = 3,
                            stride = 2 if shortcut_conv else 1,  
                            padding=1, bias= False)
        self.bn1 = BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.relu = ReLU(inplace= True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size = 3,
                            stride = 1, padding = 1, bias = False)
        self.bn2 = BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        
        if shortcut_conv:
            self.downsample = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size = 1,
                                stride = 2, padding=0, bias= False),
                BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
                )
            
        self.activate = ReLU(inplace= True)
            
    def forward(self,x):
        s = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.conv:
            s = self.downsample(s)
            
        out = self.activate(x+s)
        return out

    

class PspPooling(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,h,w)
    def __init__(self, in_channels, out_channels, 
                 feature_size=(36,100), #(h,w)
                 sizes = [1,2,3,6]):
        super().__init__()
        
        self.pools = nn.ModuleList()
        for size in sizes:#进行不同尺度的压缩后复原
            layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(size, size)),  #压缩尺寸
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.Upsample(size=feature_size, mode='bilinear'), #复原尺寸
                )
            self.pools.append(layers)
        #将上述不同尺度压缩复原的数据与原始数据共同进行卷积
        self.conv = nn.Conv2d(in_channels*(len(sizes)+1), out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):#(bs,in_channels,h,w)
        #将不同尺度压缩复原的数据与原始数据拼接
        # print(self.pools[0](x).shape)
        x = [pool(x) for pool in self.pools]+[x] #len(sizes)+1个元素，每个尺寸为#(bs,in_channels,h,w)
        # print(x[0].shape)
        # print(len(x))
        x = torch.cat(x,dim=1)
        # print(x.shape)
        x = self.conv(x) #(bs,out_channels,h,w)
        x = self.relu(x)
        return x


class PspUpsample(nn.Module):
    #(bs,in_channels,h,w)->(bs,out_channels,2h,2w)
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            )

    def forward(self, x):
        x=self.up(x)
        return x
    
class PSPNet(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)  
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 input_size = (288,800),
                 pretrained = False, #模型features部分是否使用预训练的vgg16bn模型
                 ):
        super().__init__()    
        # input_size = input_size
        '模型结构'
        ###features(encoder) #对输入数据尺寸进行压缩至输入尺寸的1/32
        if pretrained:
            #从resnet中去掉最后用于分类的两层，作为特征提取部分
            backbone = resnet18(pretrained)
            backbone = list(backbone.children())[:-2]
            

            
            #将主干部分分为两段，提取一个中间输入
            # size/16 | channel:3->256
            self.features1 = nn.Sequential(*backbone[:-1]) 
            # size/2 | channel:256->512
            self.features2 = nn.Sequential(*backbone[-1:])
                
                
        else:
            # size/16 | channel:3->256
            self.features1 = nn.Sequential(
                # size/2 | channel:3->64
                Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                BatchNorm2d(64, eps=1e-05, momentum=0.1),
                ReLU(inplace=True),
                
                # size/2 | channel:64
                MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                
                # size/2 | channel:64->64
                BasicBlock(64, 64, shortcut_conv=False),
                BasicBlock(64, 64, shortcut_conv=False),
                
                # size/2 | channel:64->128
                BasicBlock(64, 128, shortcut_conv=True),
                BasicBlock(128, 128, shortcut_conv=False),
                
                # size/2 | channel:128->256
                nn.Sequential(BasicBlock(128, 256, shortcut_conv=True),
                              BasicBlock(256, 256, shortcut_conv=False)),  
                )
            
            self.features2 = nn.Sequential(                
                # size/2 | channel:256->512
                nn.Sequential(BasicBlock(256, 512, shortcut_conv=True),
                              BasicBlock(512, 512, shortcut_conv=False)),    
                )
    
         
        #替换resnet的后两个basicblock中的卷积，stride变为1，dilation变为2和4         
        #替换后features1: size/8 | channel:3->256
        self.features1[-1][0].conv1 = Conv2d(128, 256, 
                                       kernel_size=3, stride=1, dilation = 2, 
                                       padding = 2, bias=False)
        self.features1[-1][0].downsample[0] = Conv2d(128, 256, 
                                       kernel_size=1, stride=1, bias=False)
        
        #替换后features2: size== | channel:256->512
        self.features2[-1][0].conv1 = Conv2d(256, 512, 
                                       kernel_size=3, stride=1, dilation = 4, 
                                       padding = 4, bias=False)
        self.features2[-1][0].downsample[0] = Conv2d(256, 512, 
                                       kernel_size=1, stride=1, bias=False)        

             
        feature_scale = 8#由于两个使尺寸/2的层stride变为1，因此总尺寸变化为1/8
        feature_size = [i//feature_scale for i in input_size] #通过features处理的数据尺寸变为原输入的1/8
        # size== | channel:512->1024
        self.pyramid = PspPooling(512, 1024, feature_size)
        self.drop = nn.Dropout2d(p=0.3)
        
        
        # size*8 | channel:1024->64 
        self.up = nn.Sequential(
            # size*2| channel:1024->256
            PspUpsample(1024, 256),
            nn.Dropout2d(p=0.1),
            
            # size*2 | channel:256->64
            PspUpsample(256, 64),
            nn.Dropout2d(p=0.1),
            
            # size*2 | channel:64->64
            PspUpsample(64, 64),
            nn.Dropout2d(p=0.1),
            )
  
        # size== | channel:64->n_cls
        self.final_conv = nn.Conv2d(64, n_cls, kernel_size=1)
        
        # size== | channel:256->n_cls
        self.aux_conv = nn.Conv2d(256, n_cls, kernel_size=1)
        #features的输出尺寸为1/8，还原需要*8
        # size*8 | channel==
        self.aux_up = nn.Upsample(scale_factor=8, mode='bilinear')

            
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()

    def forward(self,x, auxiliary=True): #x (bs,3,h,w)

        aux = self.features1(x) #aux (bs,256,/8)
        x = self.features2(aux) #x (bs,512,/8)

        x = self.pyramid(x) #x (bs,1024,/8)
        x = self.drop(x)
        
        x = self.up(x) #x (bs,64,-)
        x = self.final_conv(x) #x (bs,n_cls,-)
        
        if auxiliary: #训练时需要aux结果
            aux = self.aux_conv(aux) #aux (bs,n_cls,/8)
            aux = self.aux_up(aux) #aux (bs,n_cls,-)           
            return x, aux
        return x
    
    def loss(self, pred, targ):
        #pred包含两个双元素，x与aux的结果，形状均为(bs,n_cls,h,w)
        #targ为分割标签，(bs,h,w)，值为0~n_cls-1
        
        seg_pred, aux_pred = pred
        
        loss = self.celoss(seg_pred,targ) + 0.3*self.celoss(aux_pred,targ)
        return loss
    

if __name__ == '__main__':
    inputs = torch.zeros(1,3,224,224)
    n_cls = 5
    input_size = inputs.shape[2:]
    
    for pretrained in [True,False]:
        model = PSPNet(n_cls,input_size,pretrained)
        outputs = model(inputs)
        print(f'classes: {n_cls} || pretrained: {pretrained} ')
        print(f'inputshape:{inputs.shape} || outputshape:{[i.shape for i in outputs]}\n')


#自写的仿结构的低参数量模型
class PSPNet_small(nn.Module):
    #(bs,3,h,w)->(bs,n_cls,h,w)  
    def __init__(self, 
                 n_cls=11, #进行图像分割的总类别数（包含背景）
                 input_size = (288,800),
                 ):
        super().__init__()    
        # input_size = input_size
        '模型结构'
        ###features(encoder) #对输入数据尺寸进行压缩至输入尺寸的1/32
        # size/16 | channel:3->256
        self.features1 = nn.Sequential(
            # size/2 | channel:3->64
            Conv2d(3, 8, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            BatchNorm2d(8, eps=1e-05, momentum=0.1),
            ReLU(inplace=True),
            
            # size/2 | channel:64
            MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            
            # size/2 | channel:64->64
            BasicBlock(8, 8, shortcut_conv=False),
            BasicBlock(8, 8, shortcut_conv=False),
            
            # size/2 | channel:64->128
            BasicBlock(8, 16, shortcut_conv=True),
            BasicBlock(16, 16, shortcut_conv=False),
            
            # size/2 | channel:128->256
            nn.Sequential(BasicBlock(16, 32, shortcut_conv=True),
                          BasicBlock(32, 32, shortcut_conv=False)),  
            )
        
        self.features2 = nn.Sequential(                
            # size/2 | channel:256->512
            nn.Sequential(BasicBlock(32, 64, shortcut_conv=True),
                          BasicBlock(64, 64, shortcut_conv=False)),    
            )
    
         
        #替换resnet的后两个basicblock中的卷积，stride变为1，dilation变为2和4         
        #替换后features1: size/8 | channel:3->256
        self.features1[-1][0].conv1 = Conv2d(16, 32, 
                                       kernel_size=3, stride=1, dilation = 2, 
                                       padding = 2, bias=False)
        self.features1[-1][0].downsample[0] = Conv2d(16, 32, 
                                       kernel_size=1, stride=1, bias=False)
        
        #替换后features2: size== | channel:256->512
        self.features2[-1][0].conv1 = Conv2d(32, 64, 
                                       kernel_size=3, stride=1, dilation = 4, 
                                       padding = 4, bias=False)
        self.features2[-1][0].downsample[0] = Conv2d(32, 64, 
                                       kernel_size=1, stride=1, bias=False)        

             
        feature_scale = 8#由于两个使尺寸/2的层stride变为1，因此总尺寸变化为1/8
        feature_size = [i//feature_scale for i in input_size] #通过features处理的数据尺寸变为原输入的1/8
        # size== | channel:512->1024
        self.pyramid = PspPooling(64, 128, feature_size)
        self.drop = nn.Dropout2d(p=0.3)
        
        
        # size*8 | channel:1024->64 
        self.up = nn.Sequential(
            # size*2| channel:1024->256
            PspUpsample(128, 64),
            nn.Dropout2d(p=0.1),
            
            # size*2 | channel:256->64
            PspUpsample(64, 32),
            nn.Dropout2d(p=0.1),
            
            # size*2 | channel:64->64
            PspUpsample(32, 32),
            nn.Dropout2d(p=0.1),
            )
  
        # size== | channel:64->n_cls
        self.final_conv = nn.Conv2d(32, n_cls, kernel_size=1)
        
        # size== | channel:256->n_cls
        self.aux_conv = nn.Conv2d(32, n_cls, kernel_size=1)
        #features的输出尺寸为1/8，还原需要*8
        # size*8 | channel==
        self.aux_up = nn.Upsample(scale_factor=8, mode='bilinear')

            
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss()

    def forward(self,x, auxiliary=True): #x (bs,3,h,w)

        aux = self.features1(x) #aux (bs,256,/8)
        x = self.features2(aux) #x (bs,512,/8)

        x = self.pyramid(x) #x (bs,1024,/8)
        x = self.drop(x)
        
        x = self.up(x) #x (bs,64,-)
        x = self.final_conv(x) #x (bs,n_cls,-)
        
        if auxiliary: #训练时需要aux结果
            aux = self.aux_conv(aux) #aux (bs,n_cls,/8)
            aux = self.aux_up(aux) #aux (bs,n_cls,-)           
            return x, aux
        return x
    
    def loss(self, pred, targ):
        #pred包含两个双元素，x与aux的结果，形状均为(bs,n_cls,h,w)
        #targ为分割标签，(bs,h,w)，值为0~n_cls-1
        
        seg_pred, aux_pred = pred
        
        loss = self.celoss(seg_pred,targ) + 0.3*self.celoss(aux_pred,targ)
        return loss