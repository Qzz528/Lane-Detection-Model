# -*- coding: utf-8 -*-
"""
图片和分割标签的数据集
"""

from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from data_augment import segResize, segToTensor
from data_augment import simCompose
from data_augment import RandomRotate, RandomTranslationH, RandomTranslationV

#据此生成图片变换方法：对图片数据进行尺寸变换与tensor张量转换
img_transform_generate = lambda h,w: Compose([Resize((h, w)),#尺寸变换
                                              ToTensor(), #转换为张量，同时像素值范围由0~255变为0~1
                                              # Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)), #标准化
                                              ])
#据此生成分割标签变换方法：对分割标签数据进行尺寸变换并转为tensor张量
seg_transform_generate = lambda h,w:Compose([segResize((h, w)),#尺寸变换
                                             segToTensor(), #转换为张量
                                             ])

#定义数据增强方式，同时对输入图片和分割标签进行处理
data_augment = simCompose([RandomRotate(6),        #旋转
                           RandomTranslationV(100),#竖移
                           RandomTranslationH(200),#平移
                           ])

#定义数据集
class CULane_ImgSeg(Dataset):  #图片，分割标签
    def __init__(self, 
                 data_path, #数据集位置
                 data_size = (288,800), #输入的图片尺寸（输出的分割结果尺寸）
                 data_type = 'train', #训练集或验证集 'train' 或 'val'
                 augment = False, #是否进行数据增强
                 ):
        super().__init__()
        data_list = data_path + r"/list/" + data_type + "_gt.txt"
        #读取文件列表
        with open(data_list, 'r') as f:
            self.data_file = f.readlines()          
        self.data_path = data_path

        self.h,self.w = data_size
        self.img_trans = img_transform_generate(self.h,self.w)
        self.seg_trans = seg_transform_generate(self.h,self.w)
        
        self.aug_trans = data_augment if augment else None
        
    def __getitem__(self, index):#从数据库取第index条数据
        #txt文件中取第index行，并拆分，获取原图和分割标签的路径
        file_info = self.data_file[index].split(' ')
        img_name, seg_name = file_info[0], file_info[1] 

        #获取原图和分割标签的绝对路径    
        img_path = self.data_path + img_name
        seg_path = self.data_path + seg_name
        #读取数据
        img = Image.open(img_path)                
        seg = Image.open(seg_path)

        #对图片和分割标签做同样的平移和旋转，扩充数据集，数据增强
        if self.aug_trans:
            img, seg = self.aug_trans(img, seg)

        #对图片和分割标签分别进行变换，使其尺寸对应模型的输入输出
        img = self.img_trans(img) #图片(3,h,w)
        seg = self.seg_trans(seg) #分割标签(h,w)    
        return img, seg

    def __len__(self):#数据库长度
        return len(self.data_file)    
    


class CULane_ImgSegExi(Dataset): #图片，分割标签，车道线存在标签
    def __init__(self, 
                 data_path, #数据集位置
                 data_size = (288,800), #输入的图片尺寸（输出的分割结果尺寸）
                 data_type = 'train', #训练集或验证集 'train' 或 'val'
                 augment = False, #是否进行数据增强
                 ):
        super().__init__()
        data_list = data_path + r"/list/" + data_type + "_gt.txt"
        #读取文件列表
        with open(data_list, 'r') as f:
            self.data_file = f.readlines()          
        self.data_path = data_path

        self.h,self.w = data_size
        self.img_trans = img_transform_generate(self.h,self.w)
        self.seg_trans = seg_transform_generate(self.h,self.w)
        
        self.aug_trans = data_augment if augment else None
        
    def __getitem__(self, index):#从数据库取第index条数据
        #文件中取第index行，并拆分，获取原图和分割标签的路径
        file_info = self.data_file[index].split(' ')
        img_name, seg_name = file_info[0], file_info[1] 
        #文件中 四条车道线是否存在的标签 存在值为1不存在为0
        exi= np.array([int(file_info[i]) for i in range(2,6)])
        
        #获取原图和分割标签的绝对路径    
        img_path = self.data_path + img_name
        seg_path = self.data_path + seg_name
        #读取数据
        img = Image.open(img_path)                
        seg = Image.open(seg_path) 
        
        #对图片和分割标签做同样的平移和旋转，扩充数据集，数据增强
        if self.aug_trans:
            img, seg = self.aug_trans(img, seg)

        #对图片和分割标签分别进行变换，使其尺寸对应模型的输入输出
        img = self.img_trans(img) #图片，尺寸(3,h,w)，浮点0~1
        seg = self.seg_trans(seg) #分割标签，尺寸(h,w)，整型0~4
        exi = torch.from_numpy(exi).float() #四条车道线是否存在，尺寸(4)，0或1
        
        return img, seg, exi

    def __len__(self):#数据库长度
        return len(self.data_file)    
    
    
if __name__ == '__main__':
    
    data_path = r"F:/CULane" #TODO:数据集的实际地址

    print("Image & Segment:")
    #创建训练集和测试集    
    trainset = CULane_ImgSeg(data_path,(288,800),'train')
    img,seg = trainset[0]#取训练集的第一组数据
    print(img.shape ,seg.shape) #图片尺寸，分割标签尺寸
    #=> torch.Size([3, 288, 800]) torch.Size([288, 800])
    
    #将数据集组成dataloader，提供数据供模型使用
    trainloader = DataLoader(trainset, batch_size = 16, shuffle = True)
    for img,seg in trainloader:
        print(img.shape, seg.shape) #一批图片和分割标签的数据
        #=> torch.Size([16, 3, 288, 800]) torch.Size([16, 288, 800])
        break
    
    print("Image & Segment & Existance:")
    #创建训练集和测试集    
    trainset = CULane_ImgSegExi(data_path,(288,800),'train')
    img,seg,exi = trainset[0]#取训练集的第一组数据
    print(img.shape ,seg.shape, exi.shape) #图片尺寸，分割标签尺寸，车道线存在标签尺寸
    #=> torch.Size([3, 288, 800]) torch.Size([288, 800]) torch.Size([4])
    
    #将数据集组成dataloader，提供数据供模型使用
    trainloader = DataLoader(trainset, batch_size = 16, shuffle = True)
    for img,seg,exi in trainloader:
        print(img.shape, seg.shape, exi.shape) #一批图片、分割标签、车道线存在的数据
        #=> torch.Size([16, 3, 288, 800]) torch.Size([16, 288, 800]) torch.Size([16, 4])
        break
    