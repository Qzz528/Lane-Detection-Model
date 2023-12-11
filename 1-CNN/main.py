# -*- coding: utf-8 -*-
'定义模型'
#导入模型，数据集，评估方法（确保文件夹下有对应的py文件）
from dataset import CULaneData, img_transform
from model import CNN
from utils import get_metric, get_score
#导入库
from PIL import Image
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt  
from tqdm import tqdm
from torch.utils.data import DataLoader
print('[Import Complete]')
#创建模型
model = CNN() 
model_path= r"Simple-CNN.pt" #TODO：请修改为进行模型存储的实际地址

try: #如果本地已有同名模型，则载入模型继续训练
    model.load_state_dict(torch.load(model_path))
    print('[Train from Checkpoint]')
except: #否则从头训练
    print('[Train from Startpoint]')    
#设置训练设备，并把模型加载到设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#%%
'训练模型'
data_path = r"F:/CULane" #TODO：请修改为数据集的实际地址
#超参数
epoch = 15 #训练轮数
learning_rate = 1e-3 #学习率
batch_size = 32 #批尺寸 

#创建训练集和测试集    
trainset = CULaneData(data_path,'train')
valset = CULaneData(data_path,'val')
#根据数据集设置dataloader（分别为训练，验证用）
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
valloader = DataLoader(valset, batch_size = batch_size, shuffle = False)

#优化器
optimizer = optim.Adam(lr = learning_rate, params = model.parameters())
#损失函数（针对于每个像素点的多分类问题）
loss_f = CrossEntropyLoss(reduction = 'mean')

#训练及验证
for i in range(epoch):#每轮
    #训练
    model.train() 
    loss_train = 0 #训练总损失
    metric_train = np.array([0,0,0]) #训练总统计 #分别为TP,FN,FP
    trainbar =  tqdm(trainloader,desc = f"[Train Epoch {i}]") #设置进度条
    for j,(img,seg) in enumerate(trainbar):#取训练数据
        img = img.to(device) #(bs,3,h,w)
        seg = seg.to(device) #(bs,5,h,w)

        #调用模型输出分割结果
        pred = model(img)  #(bs,5,h,w)
        loss = loss_f(pred,seg) #计算模型结果与标签（正确结果）的偏差，该批次的损失
        loss_train += loss.item() #累加计入训练总损失

        loss.backward() #梯度反向传播
        optimizer.step() #更新模型参数 
        optimizer.zero_grad() #清空梯度

        pred = pred.detach().cpu().numpy()  
        seg = seg.cpu().numpy()
        #每个像素点处有5个值，其中最大值对应的通道序号是该点的车道线类别（四种车道和无车道）
        pred = np.argmax(pred,axis=1) #(bs,h,w)
        metric = get_metric(pred, seg, iou_thresh = 0.5) #该批次的统计
        metric_train += np.array(metric) #累加计入总统计

        trainbar.set_postfix({'loss':loss.item()}) #进度条显示该批的实时损失

        ###########展示（如不需要可删除该部分）
        if j%1000 == 0: #每训练1000批，进行一次图片展示
            img, seg = trainset[1] #从验证集取一条数据
            #调用模型
            pred = model(img[None,:].to(device))[0] #(5,h,w)
            pred = pred.detach().cpu().numpy()  
            pred = np.argmax(pred,axis=0) #(5,h,w)->(h,w)
            #显示图片
            img = img.permute(1,2,0).cpu().numpy()
            img = (img*255).astype('uint8')
            plt.imshow(img)
            plt.show()
            #显示真实分割结果
            seg = seg.cpu().numpy()
            plt.matshow(seg)
            plt.show()
            #显示模型输出的分割结果
            fig, ax = plt.subplots()
            ax.text(0.9, 0.9, f'epoch{i}-batch{j}', ha = 'right', va = 'top', transform=ax.transAxes, color = 'w', size = 20)
            ax.matshow(pred)
            # plt.savefig(f'{i}-{j}.png') #存储图片
            plt.show()      
        ##############

    tLoss = loss_train/len(trainloader.dataset) #每轮结束打印该轮总损失
    tPer,tRec,tF1 = get_score(*metric_train) #每轮结束打印该轮总指标
    print("[Train SUMMARY]",
          "Loss:%.5f,"%(tLoss),
          "Percision:%.2f%%,"%(tPer*100),
          "Recall:%.2f%%,"%(tRec*100),
          "F1Score:%.2f%%"%(tF1*100)) 

    #验证
    model.eval()
    loss_val = 0 #验证总损失
    metric_val = np.array([0,0,0]) #验证总统计结果 #分别为TP,FN,FP
    valbar =  tqdm(valloader,desc = f"[Eval Epoch {i}]") #设置进度条
    for j,(img,seg) in enumerate(valbar):#取验证数据
        img = img.to(device) #(bs,3,h,w)
        seg = seg.to(device) #(bs,5,h,w)

        #调用模型输出分割结果
        pred = model(img) #(bs,5,h,w)
        loss = loss_f(pred,seg) #计算模型结果与标签（正确结果）的偏差
        loss_val += loss.item() #累加验证总损失

        pred = pred.detach().cpu().numpy()  
        seg = seg.cpu().numpy()
        #每个像素点处有5个值，其中最大值对应的通道序号是该点的车道线类别（四种车道和无车道）
        pred = np.argmax(pred,axis=1) #(bs,h,w)
        metric = get_metric(pred, seg, iou_thresh = 0.5) #该批次的统计结果
        metric_val += np.array(metric) #累加计入总统计结果

        valbar.set_postfix({'loss':loss.item()}) #进度条显示该批的实时损失

    tLoss = loss_val/len(trainloader.dataset) #每轮结束打印该轮总损失
    tPer,tRec,tF1 = get_score(*metric_val) #每轮结束打印该轮总指标
    print("[Eval SUMMARY]",
          "Loss:%.5f,"%(tLoss),
          "Percision:%.2f%%,"%(tPer*100),
          "Recall:%.2f%%,"%(tRec*100),
          "F1Score:%.2f%%\n"%(tF1*100))  

    #保存模型到本地
    torch.save(model.state_dict(), model_path) 


#%%
'调用模型' 
#读取一张图片 
image_path = "sample.jpg" #TODO:使用时修改为要进行车道检测的实际图片地址
img = Image.open(image_path) 
#将其用训练时同样的图片变换函数进行处理，以使其变为张量，范围变为0~1，并符合模型输入尺寸
img = img_transform(img).to(device)

#使用模型
pred = model(img[None,:])[0] #(3,h,w)->(1,3,h,w)=model=>(1,5,h,w)->(5,h,w)

#每个像素点处有5个值，其中最大值对应的通道序号是该点的车道线类别（四种车道和无车道）
pred = pred.detach().cpu().numpy()  
pred = np.argmax(pred,axis=0) #(5,h,w)->(h,w)


#将模型输入的图片，范围由0~1还原到0~255，rgb通道由第0轴换到第2轴
img = (255*(img.cpu().numpy().transpose(1,2,0))).astype('uint8')
color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)] #对四种车道线分别上红绿蓝黄四色
for i,c in zip(range(1,5),color):#依次在原图上绘制车道线
    img[pred==i,:] = c#分割结果中为车道线的像素点，在原图上进行重新绘制为指定颜色

plt.imshow(img)
plt.show()