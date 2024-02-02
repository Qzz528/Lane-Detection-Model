# -*- coding: utf-8 -*-
"""
CULane数据集数据展示以及标签生成
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

data_path = r"F:/CULane" #TODO:修改为数据集的实际位置

##读取训练文件列表
train_list = data_path + r"/list/train_gt.txt"
with open(train_list, 'r') as f:
    train_file = f.readlines()

print(len(train_file)) #总数据条数 #=> 88880
print(train_file[0]) #第一条数据内容
#=> /driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 1 1



##将内容按空格划分，依次为图片位置，分割标签位置，4条车道线存在情况
img_name = train_file[0].split(' ')[0]#图片位置 
print(img_name)  #=> /driver_23_30frame/05151649_0422.MP4/00000.jpg

seg_name = train_file[0].split(' ')[1]#分割标签位置
print(seg_name) #=> /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png

lane_exist = train_file[0].split(' ')[2:] #4条车道线存在情况
lane_exist = [int(x) for x in lane_exist] #str转int
print(lane_exist) #=> [1, 1, 1, 1]

#读取并展示图片
img = Image.open(data_path + img_name)
img = np.array(img)
plt.imshow(img)
plt.show()
print(img.shape) #图片尺寸 #=> (590, 1640, 3)

#读取并展示分割标签
seg = Image.open(data_path + seg_name)
seg = np.array(seg)
plt.matshow(seg)
plt.colorbar()
plt.show()
print(seg.shape) #分割标签尺寸 #=> (590, 1640)

### 以上内容足可用于训练，以下为扩展内容


##分割标签实际上由与图片同名的文本文件生成
txt_name = img_name.replace('jpg','lines.txt')#同名文本文件位置 
#打开文本文件并处理，文本文件是车道线上一系列标记点的坐标
with open(data_path + txt_name, 'r') as f:
    annos = f.readlines()
annos = [lane.strip().split(' ') for lane in annos] #去除每行末的换行符再按空格分割
annos = [[float(x) for x in lane]for lane in annos] #将所有元素从str变为float

#创建坐标标签
coordinate = [[] for _ in range(4)] #创建一个嵌套列表，将标记点的横纵坐标作为一组存入
for i, lane in enumerate(annos): #i是车道线序数，lane是第i条车道线的数据
    for x,y in zip(lane[::2],lane[1::2]):#x,y是第i条车道线的横纵坐标
        coordinate[i].append((round(x),round(y))) #像素点坐标为整数，对xy进行四舍五入
#此时coordinate是一个嵌套列表，第一层四个元素，代表四条车道线
#这四个元素是list类型，其内部的第二层元素类型为tuple，是车道线各个标记点的横纵坐标
      
#对坐标点作图  
vis = img.copy() #复制一下图片，避免绘制在原图上
for lane in coordinate:#对每条车道线
    for xy in lane:#对车道线内的每个标记点
        cv2.circle(vis,xy,5,(255,0,0),thickness=-1) #作图
plt.imshow(vis)
plt.show()

#将坐标点连成线，作图
vis = img.copy() #复制一下图片，避免绘制在原图上
for lane in coordinate:#对每条车道线
    for i in range(len(lane)-1): #所有标记点两两连线 n个点，n-1条线
        cv2.line(vis,lane[i],lane[i+1],(255,0,0),thickness=5) 
plt.imshow(vis)
plt.show()

#将点连成线，做分割标签
seg = np.zeros(img.shape[:2])
for ind,lane in enumerate(coordinate):
    for i in range(len(lane)-1): #两两连线 i个点 i-1条线
        #如果是语义分割，每个车道线值均为1。如果是实例分割，车道线值为1~4
        img = cv2.line(seg,lane[i],lane[i+1],ind+1,thickness=15) 
plt.matshow(seg)
plt.colorbar()
plt.show()

##根据文本文件中的坐标点绘制的分割标签与CULane数据集原有的有所不同
##论文请使用官方的分割标签数据

##其他车道线数据集（如Tusimple）没有现成的分割标签，需用上述思路生成分割标签
##CULane每条车道线的横纵坐标混在一行数据中。
##Tusimple则是横纵坐标分开记录，车道线共用一组纵坐标。注意坐标点连线时要筛除值为-2的不存在的点
