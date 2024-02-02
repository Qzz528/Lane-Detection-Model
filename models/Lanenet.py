# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:05:46 2024

@author: Administrator
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
#Lanenet
#init instance segmatic
#特征后 cluster

# seg-> coord
# coord ->affine
# affine -> fit
# fit -> unaffine
# MSE

#[[a,b,c],[0,d,e],[0,f,1]]
#预测的6个系数分别为abcdef
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

    
def fit_coord():
    #seg->coord
    #coord -x-> coord'
    #coord' -fit-> coord_'
    #coord_' -x'-> coord_
    pass
class HNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        h,w = input_size
        
        self.feature = nn.Sequential(
            conv_bn_relu(3,16),
            conv_bn_relu(16,16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            conv_bn_relu(16,32),
            conv_bn_relu(32,32),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            conv_bn_relu(32,64),
            conv_bn_relu(64,64),
            nn.MaxPool2d(kernel_size=2, stride=2),                
            )
        
        #通道64，尺寸变为/8
        self.classifier = nn.Sequential(
            nn.Linear(h*w, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024,6),
            )
    def forward(self, x):
        
        bs = x.shape[0]
        x = self.feature(x)
        x = x.view(bs,-1)
        x = self.classifier(x)
        
        #放射变换矩阵
        T = torch.zeros(bs,3,3)
        T[:,0,0] = x[:,0] #a
        T[:,0,1] = x[:,1] #b
        T[:,0,2] = x[:,2] #c
        T[:,1,1] = x[:,3] #d
        T[:,1,2] = x[:,4] #e
        T[:,2,1] = x[:,5] #f
        T[:,2,2] = 1
        
        return T
    

post = HNet((256,256))
img = torch.ones(16,3,256,256)
trans = post(img)

bs = 2
n_lanes = 4
n_anchors = 12

xy1 = torch.ones(bs,n_lanes,n_anchors,3,1) #(bs,n_lanes,n_anchors,3,1)

xy1_ = torch.zeros(bs,n_lanes,n_anchors,3,1)
for i in range(bs):
    for j in range(n_lanes):
        for k in range(n_anchors):            
            xy1_[i,j,k] = trans[i] @ xy1[i,j,k]
            
#直接解析解
# Y = torch.zeros(bs,n_lanes,n_anchors)
# X = torch.zeros(bs,n_lanes,n_anchors)
X = xy1_[:,:,:,0]
Y = xy1_[:,:,:,1]
Y_ = torch.cat((Y**2,Y,torch.ones_like(Y)),dim=-1)

p = torch.zeros(bs,n_lanes,3,1)
for i in range(bs):
    for j in range(n_lanes):
        p[i,j] = torch.linalg.inv(Y_[i,j].T @ Y_[i,j]) @ Y_[i,j].T @ X[i,j]

X_fit = torch.zeros_like(X)    
for i in range(bs):
    for j in range(n_lanes):
        X_fit[i,j] = p[i,j,0] * (Y[i,j]**2) + (p[i,j,1] *Y[i,j]) +p[i,j,2]
        
xy1_fit = xy1_.copy()
