# -*- coding: utf-8 -*-
from PIL import Image
import torch
import numpy as np
import random
from torchvision.transforms import Compose,Resize,ToTensor
import matplotlib.pyplot as plt

#仿照torchvision中Resize和ToTensor定义segResize，segToTensor
#分别用于分割标签的尺寸变换，张量转换 
#不能和图片用同样的Resize（插值）方法，分割标签尺寸变换时需要用临近值补齐
class segResize(object): #尺寸变换
    def __init__(self,size):
        self.size = size
    def __call__(self,seg):
        return seg.resize((self.size[1], self.size[0]), Image.NEAREST)
class segToTensor(object): #张量转换 
    def __call__(self, seg):
        return torch.from_numpy(np.array(seg, dtype=np.int32)).long()#整型



##数据增强 要对图片和分割标签做同样的变换（如平移 旋转）所以内部的变换需要自写
class RandomTranslationV(object): #竖直平移
    def __init__(self,max_offset):
        self.max_offset = max_offset #最大平移距离
    def __call__(self,img,seg):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        w, h = img.size
        #平移后剩余部分补0（黑色）
        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        seg = np.array(seg)
        if offset > 0:
            seg[offset:,:] = seg[0:h-offset,:]
            seg[:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            seg[0:h-real_offset,:] = seg[real_offset:,:]
            seg[h-real_offset:,:] = 0
        return Image.fromarray(img),Image.fromarray(seg)

class RandomTranslationH(object): #水平平移
    def __init__(self,max_offset):
        self.max_offset = max_offset #最大平移距离
    def __call__(self,img,seg):
        offset = np.random.randint(-self.max_offset,self.max_offset)
        w, h = img.size
        #平移后剩余部分补0（黑色）
        img = np.array(img)
        if offset > 0:
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        seg = np.array(seg)
        if offset > 0:
            seg[:,offset:] = seg[:,0:w-offset]
            seg[:,:offset] = 0
        if offset < 0:
            real_offset = -offset
            seg[:,0:w-real_offset] = seg[:,real_offset:]
            seg[:,w-real_offset:] = 0
        return Image.fromarray(img),Image.fromarray(seg)  
    
class RandomRotate(object):#旋转

    def __init__(self, max_angle):
        self.max_angle = max_angle #最大旋转角度
    def __call__(self, img, seg):
        angle = random.randint(0, self.max_angle * 2) - self.max_angle
        img = img.rotate(angle, resample=Image.BILINEAR)
        seg = seg.rotate(angle, resample=Image.NEAREST)        
        return img, seg

#仿照torchvision中Compose定义simCompose
#用于同时对图片和分割标签进行一系列处理
class simCompose(object):
    def __init__(self, transforms): #一系列的处理方式
        self.transforms = transforms

    def __call__(self, img, seg):
        for t in self.transforms: #依次处理
            img, seg = t(img, seg)
        return img, seg
   
    
if __name__ == '__main__':
    
    data_path = r"F:/CULane" #TODO:数据集的实际地址
    ##读取训练文件列表
    train_list = data_path + r"/list/train_gt.txt"
    with open(train_list, 'r') as f:
        train_file = f.readlines()
        
    img_name = train_file[0].split(' ')[0]#图片位置 
    print(img_name)  #=> /driver_23_30frame/05151649_0422.MP4/00000.jpg

    seg_name = train_file[0].split(' ')[1]#分割标签位置
    print(seg_name) #=> /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png


    h,w = 288,800 #定义高和宽
    
    
    #创建图片变换方法：对图片数据进行尺寸变换与tensor张量转换
    img_transform = Compose([Resize((h, w)),#尺寸变换
                             ToTensor(), #转换为张量，同时像素值范围由0~255变为0~1
                             ])
    #读取图片并变换
    img = Image.open(data_path + img_name)
    img = img_transform(img)
    print(img.shape) #=> torch.Size([3, 288, 800])
    
    
    #创建分割标签变换方法：对分割标签数据进行尺寸变换并转为tensor张量
    seg_transform = Compose([segResize((h, w)),#尺寸变换
                             segToTensor(), #转换为张量
                             ])
    
    #读取分割标签并变换
    seg = Image.open(data_path + seg_name)
    seg = seg_transform(seg)
    print(seg.shape) #=> torch.Size([288, 800])
    
    
    #创建数据增强方法（同时平移旋转图片和分割标签）
    sim_transform = simCompose([RandomRotate(6),        #旋转
                                RandomTranslationV(100),#竖移
                                RandomTranslationH(200),#平移
                                ])
    #读取图片和分割标签，做同等的变换并展示
    img = Image.open(data_path + img_name)
    seg = Image.open(data_path + seg_name)
    img,seg = sim_transform(img,seg)
    
    img = np.array(img)
    plt.imshow(img)
    plt.show()

    seg = np.array(seg)
    plt.matshow(seg)
    plt.colorbar()
    plt.show()

    

    
