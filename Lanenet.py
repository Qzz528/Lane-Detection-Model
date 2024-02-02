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

# X_fit = torch.zeros_like(X)    
# for i in range(bs):
#     for j in range(n_lanes):
#         X_fit[i,j] = p[i,j,0] * (Y[i,j]**2) + (p[i,j,1] *Y[i,j]) +p[i,j,2]
        
xy1_fit = xy1_.clone()
for i in range(bs):
    for j in range(n_lanes):
        xy1_fit[i,j,:,0] = p[i,j,0] * (Y[i,j]**2) + (p[i,j,1] *Y[i,j]) +p[i,j,2]
        
xy1fit = torch.zeros_like(xy1)
for i in range(bs):
    for j in range(n_lanes):
        for k in range(n_anchors):            
            xy1fit[i,j,k] = torch.linalg.inv(trans[i]) @ xy1_fit[i,j,k]
#%%

# -*- coding: utf-8 -*-

#导入库
from PIL import Image
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt  
from tqdm import tqdm
from torch.utils.data import DataLoader
#导入模型，数据集，评估方法
from models.FCN import FCN,FCN_small
from models.SegNet import SegNet,SegNet_small
from models.UNet import UNet,UNet_small
from models.PSPNet import PSPNet,PSPNet_small
from models.SpatialCNN import SpatialCNN,SpatialCNN_small

from _datasets import CULane_ImgSeg, CULane_ImgSegExi
from _metrics import get_metric, get_score, model_info
# from _tools import 
print('[Import Complete]')

#地址
model_path= r"weights/FCN_small.pt" #TODO：请修改为进行模型存储的实际地址
data_path = r"F:/CULane" #TODO：请修改为数据集的实际地址

#参数
n_lanes = 4 #车道线类别数
epoch = 10 #训练轮数
learning_rate = 3e-4 #学习率
batch_size = 16 #批尺寸 
data_size = (288,800) #模型对应的输入图片尺寸，分割结果尺寸（据此生成训练数据）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#设置训练设备

#TODO: 创建的模型与训练集应匹配 
#model   : FCN,SegNet,UNet,PSPNet|SpatialCNN        |UFast            |LaneNet
#dataset : CULane_ImgSeg         |CULane_ImgSegExi  |CULane_ImgSegCoo |CULane_ImgSegBin

#创建模型
model = PSPNet_small(n_lanes+1) #总分类数相当于车道类别数+1（包含无车道情况）  
#创建训练集和测试集    
trainset = CULane_ImgSeg(data_path, data_size, 'train', augment = True)
valset = CULane_ImgSeg(data_path, data_size, 'val')


#尝试载入模型参数
try: #如果本地已有同名模型，则载入已有参数继续训练
    model.load_state_dict(torch.load(model_path))
    print('[Train from Checkpoint]')
except: #否则从头训练
    print('[Train from Startpoint]')    
finally: #打印模型参数，并把模型加载到设备
    model_info(model) 
    model = model.to(device)



#根据数据集设置dataloader（分别为训练，验证用）
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
valloader = DataLoader(valset, batch_size = batch_size, shuffle = False)


post = HNet((288,800)) 
#优化器
optimizer = optim.Adam(lr = learning_rate, params = post.parameters())

#%%
#训练及验证
for i in range(epoch):#每轮
    #训练
    model.train() 
    loss_train = 0 #训练总损失
    metric_train = np.array([0,0,0]) #训练总统计 #分别为TP,FN,FP
    trainbar =  tqdm(trainloader,desc = f"[Train Epoch {i}]") #设置进度条
    for j, data in enumerate(trainbar):#取训练数据
    
        #组织模型输入数据
        data = tuple([i.to(device) for i in data]) #所有数据加载到设备上
        img = data[0] #(bs,3,h,w) #第一条为图片输入
        targ = data[1:] #后续条目为标签，包含分割标签及其他标签（如有）
        targ = targ[0] if len(targ)==1 else targ #如果只有分割标签将其作为targ
        
        #调用模型输出分割结果
        pred = model(img)  #判断结果，包含分割结果及其他判断（如有）
        loss = model.loss(pred,targ) #计算模型结果与标签（正确结果）的偏差，该批次的损失
        loss_train += loss.item() #累加计入训练总损失

        #模型参数更新
        loss.backward() #梯度反向传播
        optimizer.step() #更新模型参数 
        optimizer.zero_grad() #清空梯度

        ##############模型指标统计（如训练中不需要可删除该部分）
        #组织模型输出数据
        #标签如果有多条，则第一条为分割标签(bs,h,w)
        seg_targ = targ[0] if type(targ)==tuple else targ
        #模型判断结果如果有多条，第一条为分割结果(bs,n_lanes+1,h,w)
        seg_pred = pred[0] if type(pred)==tuple else pred
        
        #数据加载到cpu以进行指标统计
        seg_pred = seg_pred.detach().cpu().numpy()  
        seg_targ = seg_targ.cpu().numpy()
        
        #每个像素点处有5个值（四种车道和无车道），其中最大值对应的通道序号是该点的车道线类别
        seg_pred = np.argmax(seg_pred,axis=1) #(bs,h,w)
        metric = get_metric(seg_pred, seg_targ, iou_thresh = 0.5) #该批次的统计
        metric_train += np.array(metric) #累加计入总统计
        ##############
        

        ###########展示（如训练中不需要可删除该部分）
        if j%100 == 0: #每训练100批，进行一次图片展示
            #从训练集或验证集取一条数据
            # data = trainset[1] 
            data = valset[1]
            img, seg_targ = data[:2] #数据前两条是图片和分割结果
            #调用模型
            pred = model(img[None,:].to(device))
            #模型判断结果如果有多条，第一条为分割结果(bs,n_lanes+1,h,w)
            seg_pred = pred[0][0] if type(pred)==tuple else pred[0]
            
            #模型输出中每个像素点有5个通道值，其中值最大的类是车道线的判断结果
            seg_pred = torch.argmax(seg_pred,axis=0) #(n_lanes+1,h,w)->(h,w) 
            #显示图片
            img = img.permute(1,2,0).cpu().numpy()
            img = (img*255).astype('uint8')
            plt.imshow(img)
            plt.show()
            #显示真实分割结果
            seg_targ = seg_targ.cpu().numpy()
            plt.matshow(seg_targ)
            plt.show()
            #显示模型输出的分割结果
            seg_pred = seg_pred.detach().cpu().numpy() 
            fig, ax = plt.subplots()
            ax.text(0.9, 0.9, f'epoch{i}-batch{j}', ha = 'right', va = 'top', transform=ax.transAxes, color = 'w', size = 20)
            ax.matshow(seg_pred)
            # plt.savefig(f'{i}-{j}.png') #存储图片
            plt.show()      
        ##############

        trainbar.set_postfix({'loss':loss.item()}) #进度条显示该批的实时损失
        
    tLoss = loss_train*batch_size/len(trainloader.dataset) #每轮结束打印该轮总损失
    tPer,tRec,tF1 = get_score(*metric_train) #每轮结束打印该轮总指标
    print("[Train SUMMARY]",
          "Loss:%.5f | "%(tLoss),
          "Percision:%.2f%% | "%(tPer*100),
          "Recall:%.2f%% | "%(tRec*100),
          "F1Score:%.2f%%"%(tF1*100)) 

    #验证
    model.eval()
    loss_val = 0 #验证总损失
    metric_val = np.array([0,0,0]) #验证总统计结果 #分别为TP,FN,FP
    valbar =  tqdm(valloader,desc = f"[Eval Epoch {i}]") #设置进度条
    for j,data in enumerate(valbar):#取验证数据
    
        #组织模型输入数据
        data = tuple([i.to(device) for i in data]) #所有数据加载到设备上
        img = data[0] #(bs,3,h,w) #第一条为图片输入
        targ = data[1:] #后续条目为标签，包含分割标签及其他标签（如有）
        targ = targ[0] if len(targ)==1 else targ #如果只有分割标签将其作为targ
        
        #调用模型输出分割结果
        pred = model(img)  #判断结果，包含分割结果及其他判断（如有）
        loss = model.loss(pred,targ) #计算模型结果与标签（正确结果）的偏差，该批次的损失
        loss_val += loss.item() #累加计入训练总损失

        #组织模型输出数据
        #标签如果有多条，则第一条为分割标签(bs,h,w)
        seg_targ = targ[0] if type(targ)==tuple else targ
        #模型判断结果如果有多条，第一条为分割结果(bs,n_lanes+1,h,w)
        seg_pred = pred[0] if type(pred)==tuple else pred
        
        #数据加载到cpu以进行指标统计
        seg_pred = seg_pred.detach().cpu().numpy()  
        seg_targ = seg_targ.cpu().numpy()
        
        #每个像素点处有5个值（四种车道和无车道），其中最大值对应的通道序号是该点的车道线类别
        seg_pred = np.argmax(seg_pred,axis=1) #(bs,h,w)
        metric = get_metric(seg_pred, seg_targ, iou_thresh = 0.5) #该批次的统计
        metric_val += np.array(metric) #累加计入总统计结果

        valbar.set_postfix({'loss':loss.item()}) #进度条显示该批的实时损失

    tLoss = loss_val*batch_size/len(trainloader.dataset) #每轮结束打印该轮总损失
    tPer,tRec,tF1 = get_score(*metric_val) #每轮结束打印该轮总指标
    print("[Eval SUMMARY]",
          "Loss:%.5f | "%(tLoss),
          "Percision:%.2f%% | "%(tPer*100),
          "Recall:%.2f%% | "%(tRec*100),
          "F1Score:%.2f%%\n"%(tF1*100))  

    #保存模型到本地
    torch.save(model.state_dict(), model_path) 


