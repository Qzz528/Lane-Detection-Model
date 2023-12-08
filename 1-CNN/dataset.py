# -*- coding: utf-8 -*-
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose,Resize,ToTensor

#模型输入的图片（输出的分割结果）尺寸
h=288 #高
w=800 #宽

#创建图片变换函数：对图片数据进行尺寸变换与tensor张量转换
img_transform = Compose([Resize((h, w)),#尺寸变换
                         ToTensor(), #转换为张量，同时像素值范围由0~255变为0~1
                         ])

#仿照Resize和ToTensor定义用于分割标签的尺寸变换，张量转换 
#不能和图片用同样的Resize（插值）方法，分割标签尺寸变换时需要用临近值补齐
class segResize(object):
    def __init__(self,size):
        self.size = size
    def __call__(self,seg):
        return seg.resize((self.size[1], self.size[0]), Image.NEAREST)
class segToTensor(object):
    def __call__(self, seg):
        return torch.from_numpy(np.array(seg, dtype=np.int32)).long()#整型
#创建分割标签变换函数：对分割标签数据进行尺寸变换并转为tensor张量
seg_transform = Compose([segResize((h, w)),#尺寸变换
                         segToTensor(), #转换为张量
                         ])


#定义数据集
class CULaneData(Dataset): 
    def __init__(self, data_path, data_type = 'train'): #train or val
        super().__init__()
        data_list = data_path + r"/list/" + data_type + "_gt.txt"
        #读取文件列表
        with open(data_list, 'r') as f:
            self.data_file = f.readlines()          
        self.data_path = data_path

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
        #对图片和分割标签进行变换，使其尺寸对应模型的输入输出
        img = img_transform(img)
        seg = seg_transform(seg)
        return img, seg

    def __len__(self):#数据库长度
        return len(self.data_file)    
    
    
if __name__ == '__main__':
    
    data_path = r"F:/CULane" #TODO:数据集的实际地址

    #创建训练集和测试集    
    trainset = CULaneData(data_path,'train')
    valset = CULaneData(data_path,'val')
    img,seg = trainset[0]#取训练集的第一组数据
    print(img.shape ,seg.shape) #图片尺寸，分割标签尺寸
    #=> torch.Size([3, 288, 800]) torch.Size([288, 800])
    
    #将数据集组成dataloader，提供数据供模型使用
    trainloader = DataLoader(trainset, batch_size = 16, shuffle = True)
    valloader = DataLoader(valset, batch_size = 16, shuffle = False)
    for img,seg in trainloader:
        print(img.shape, seg.shape) #一批图片和分割标签的数据
        #=> torch.Size([16, 3, 288, 800]) torch.Size([16, 288, 800])
        break