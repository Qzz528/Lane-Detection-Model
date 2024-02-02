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

#优化器
optimizer = optim.Adam(lr = learning_rate, params = model.parameters())


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


