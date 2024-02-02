# -*- coding: utf-8 -*-
'''
参考https://github.com/harryhan618/SCNN_Pytorch 
'''

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
import torch.nn.functional as F

class SpatialCNN(nn.Module):
    #(bs,3,h,w)(bs,n_cls,h,w) & (bs,n_cls-1) 
    def __init__(self, 
                 n_cls = 5, #进行车道分割的总类别数（包含背景，值为车道数+1）
                 input_size=(288,800), #输入图片，输出分割结果的尺寸(h,w)
                 pretrained = False, #模型features部分是否使用预训练的vgg16模型
                 ks = 9, #DURL层卷积核大小
                 ):

        super().__init__()
        
        h, w = input_size
        n_lanes = n_cls-1 
        '模型结构'
        ###features #对输入数据尺寸进行压缩至输入尺寸的1/8
        self.backbone = vgg16_bn(pretrained).features #使用vgg16的部分作为主干     
        
        #1->1/8
        # for i in [34, 37, 40]:# 将部分卷积层，改用更大的dilation，提高感受野
        #     conv = self.backbone._modules[str(i)]
        #     # 通过调整padding，并不影响通过卷积层的数据尺寸
        #     dilated_conv = nn.Conv2d(
        #         conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
        #         padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
        #     )
        #     dilated_conv.load_state_dict(conv.state_dict())
        #     self.backbone._modules[str(i)] = dilated_conv

        # 去掉最后2个maxpooling，保留3个/2的maxpooling，数据变为原尺寸1/8
        self.backbone._modules.pop('33')
        self.backbone._modules.pop('43')        
        
        
        ###pre-DURL-post 改变数据通道数，通过durl模块，再改变通道数
        #channel: 512->128
        self.preConv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU() 
            )
        
        #channel: 128->128
        self.DURL = nn.ModuleList()
        # 向下向上逐行卷积，行padding=0，行kernel=1
        self.DURL+=[nn.Conv2d(128, 128, (1, ks), padding=(0,ks//2), bias=False)] #D
        self.DURL+=[nn.Conv2d(128, 128, (1, ks), padding=(0,ks//2), bias=False)] #U
        # 向右向左逐列卷积，列padding=0，列kernel=1
        self.DURL+=[nn.Conv2d(128, 128, (ks, 1), padding=(ks//2,0), bias=False)] #R
        self.DURL+=[nn.Conv2d(128, 128, (ks, 1), padding=(ks//2,0), bias=False)] #L        
        
        #channel: 128->n_cls
        self.postConv = nn.Sequential(nn.Dropout2d(0.1),
                                      nn.Conv2d(128, n_cls, 1)) 
        #postConv输出为分割结果，通过*8的插值还原到原尺寸
        
        
        ###classifier 对车道线有无进行分类的辅助部分
        #1/8->1/16
        self.pool = nn.Sequential(nn.Softmax(dim=1),  
                                  nn.AvgPool2d(2, 2) )       
        self.n_fc = n_cls * int(h/16) * int(w/16) #将此时数据展开为1维时的数据尺寸
        #->n_lanes
        self.fc = nn.Sequential(nn.Linear(self.n_fc, 128),
                                nn.ReLU(),
                                nn.Linear(128, n_lanes),
                                nn.Sigmoid()
                                )
        
        '损失函数'
        #图像分割是逐像素的多分类问题
        self.celoss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1, 1, 1, 1]))
        #当像素点无车道线时损失值权重0.4（无车道线情况比较多，适当降低权重）
        
        #对于图中每条车道线有无的二分类问题
        self.bceloss = nn.BCELoss()
        
    def allDURL(self, x): #依次运行DURL模块中的四个卷积（分别为向下，向上，向右，向左逐层卷积）
        for DURL_conv, direc in zip(self.DURL, 'DURL'):
            x = self.singleDURL(x, DURL_conv, direc)
        return x

    def singleDURL(self, x, conv, direction):#运行DURL中的某一个卷积
        #conv为要使用的卷积层，direction为逐层卷积的方向（DURL）
        _, C, H, W = x.shape
        if direction in ['D','U']: #如果向下或者向上逐层卷积
            slices = [x[:, :, i:(i + 1), :] for i in range(H)] #按行H切片
            dim = 2 #行对应的维度为2
        elif direction in ['R','L']: #如果向右或者向左逐层卷积
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)] #按列W切片
            dim = 3 #列对应的维度为3
        
        if direction in ['U','L']: #如果是从下到上、从右向左，把所有切片倒序
            slices = slices[::-1]

        out = [slices[0]] #输出的第一层是第一张切片
        for i in range(1, len(slices)): #第i层结果是第i张切片 加 第i-1层结果的卷积
            out.append(slices[i] + F.relu(conv(out[i - 1])))
            
        if direction in ['U','L']: #如果是从下到上、从右向左，计算完成后需要再把结果把所有层结果倒序
            out = out[::-1]
            
        return torch.cat(out, dim=dim) #按行或者列把所有层结果拼接


    def forward(self, x): #(bs,3,h,w)

        x = self.backbone(x) #(bs,512,h/8,w/8)
        x = self.preConv(x) #(bs,128,h/8,w/8)
        x = self.allDURL(x) #(bs,128,h/8,w/8)
        x = self.postConv(x) #(bs,n_cls,h/8,w/8)
        
        #插值上采样将数据尺寸扩大至8倍
        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        #seg_pred:(bs,n_cls,h,w)
        #输出与输入图同尺寸，每个像素有n_cls个通道值，代表n_cls类情况
        
        x = self.pool(x) #(bs,n_cls,h/16,w/16)
        x = x.view(-1, self.n_fc) #(bs,n_cls*(h/16)*(w/16))
        exi_pred = self.fc(x) #(bs,n_lanes)
        #exi_pred:(bs,n_lanes)
        #整张图内车道线存在结果（4种车道线存在的概率）
        return seg_pred, exi_pred
    
    def loss(self, pred, targ):
        #pred与trag均包含两个双元素，车道线分割以及车道线有无
        #pred模型判断结果：seg_pred车道线分割结果，exi_pred车道线有无判断结果
        seg_pred, exi_pred = pred #seg_pred:(bs,n_lanes+1,h,w) | exi_pred:(bs,h,w)
        #targ真实标签结果：seg_targ分割标签，exi_targ车道线有无标签
        seg_targ, exi_targ = targ #seg_targ:(bs,n_lanes) | exi_targ : (bs,n_lanes)
        
        seg_loss = self.celoss(seg_pred,seg_targ) #计算模型分割结果与分割标签（正确结果）的偏差
        exi_loss = self.bceloss(exi_pred,exi_targ) #计算模型判断的车道线是否存在的损失偏差
        
        loss = seg_loss + 0.1*exi_loss #综合上述两种损失，作为该批次的损失
        
        return loss


if __name__ == '__main__':
    inputs = torch.zeros(1,3,224,224)
    n_cls = 5
    input_size = inputs.shape[2:]
    
    for pretrained in [True,False]:
        for ks in [5,9]:
            model = SpatialCNN(n_cls,input_size,pretrained,ks)
            outputs = model(inputs)
            print(f'classes: {n_cls} || pretrained: {pretrained} || ks: {ks}')
            print(f'inputshape:{inputs.shape} || outputshape:{[i.shape for i in outputs]}\n')
    ##参数ks是DURL模块卷积层的卷积核大小，需要是奇数，才可保证DURL卷积层数据尺寸不变
    ##该模型把图片尺寸（宽高）压缩至原来的1/16
    ##如果图片尺寸不是16的整数倍，会导致展开时全连接层神经元个数计算错误

#自写的仿结构的低参数量模型    
class conv_bn_relu(nn.Module):
    #(bs,in_channels,h,w)(bs,out_channels,h,w)
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
    
class SpatialCNN_small(nn.Module):
    def __init__(self, 
                 n_cls = 5,#进行车道分割的总类别数（包含背景，值为车道数+1）
                 input_size=(288,800), #输入图片，输出分割结果的尺寸(h,w)
                 ks = 5,##DURL层卷积核大小
                 ):

        super().__init__()
        h, w = input_size
        
        ###features(encoder) 
        self.encoder = nn.Sequential(
            
            conv_bn_relu(3, 8),
            conv_bn_relu(8, 8),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_bn_relu(8, 16),
            conv_bn_relu(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            conv_bn_relu(16, 32),
            conv_bn_relu(32, 32),
            conv_bn_relu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            )

        ###pre-DURL-post
        self.preConv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU() 
            )
        
        self.DURL = nn.ModuleList()
        self.DURL+=[nn.Conv2d(8, 8, (1, ks), padding=(0,ks//2), bias=False)] #D
        self.DURL+=[nn.Conv2d(8, 8, (1, ks), padding=(0,ks//2), bias=False)] #U
        self.DURL+=[nn.Conv2d(8, 8, (ks, 1), padding=(ks//2,0), bias=False)] #R
        self.DURL+=[nn.Conv2d(8, 8, (ks, 1), padding=(ks//2,0), bias=False)] #L        
        
        self.postConv = nn.Sequential(nn.Dropout2d(0.1),
                                      nn.Conv2d(8, n_cls, 1)) 

        ###classifier
        self.pool = nn.Sequential(nn.Softmax(dim=1),  
                                  nn.AvgPool2d(2, 2) )
        
        self.n_fc = n_cls* int(h/16) * int(w/16) #将此时数据展开为1维时的数据尺寸
        self.fc = nn.Sequential(nn.Linear(self.n_fc, 128),
                                nn.ReLU(),
                                nn.Linear(128, n_cls-1),
                                nn.Sigmoid() )
        
        
        self.celoss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1, 1, 1, 1]))
        self.bceloss = nn.BCELoss()
        
    def allDURL(self, x): 
        for DURL_conv, direc in zip(self.DURL, 'DURL'):
            x = self.singleDURL(x, DURL_conv, direc)
        return x

    def singleDURL(self, x, conv, direction):
        _, C, H, W = x.shape
        if direction in ['D','U']: 
            slices = [x[:, :, i:(i + 1), :] for i in range(H)] 
            dim = 2 
        elif direction in ['R','L']: 
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)] 
            dim = 3 
        
        if direction in ['U','L']: 
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)): 
            out.append(slices[i] + F.relu(conv(out[i - 1])))
            
        if direction in ['U','L']: 
            out = out[::-1]
            
        return torch.cat(out, dim=dim)


    def forward(self, x):
        #size:1->1/8
        x = self.encoder(x) 
        
        #channel:->n_cls
        x = self.preConv(x) 
        x = self.allDURL(x) 
        x = self.postConv(x)
        
        #size:1/8->1
        seg_pred = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
 
        #->n_lanes
        x = self.pool(x) 
        x = x.view(-1, self.n_fc)
        exi_pred = self.fc(x) 

        return seg_pred, exi_pred
    
    def loss(self, pred, targ):
        seg_pred, exi_pred = pred 
        seg_targ, exi_targ = targ 
        
        seg_loss = self.celoss(seg_pred,seg_targ) 
        exi_loss = self.bceloss(exi_pred,exi_targ) 
        
        loss = seg_loss + 0.1*exi_loss 
        
        return loss
