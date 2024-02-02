# -*- coding: utf-8 -*-
"""
分割标签和分割坐标的生成与修正
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt  

#分割与标签互换时所依靠的锚点，适用于图片尺寸(288,800)
CULane_anchors = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
##锚点本质上是一系列纵坐标，计算出车道线在这些纵坐标处的横坐标，可获得车道线的一系列坐标点 


def seg2coord(seg, anchors,  n_lanes=4):
    '''
    将分割标签（或分割结果）转变为坐标
    
    seg: numpy.array, 分割图, shape(h,w), 值0~n_lanes
    anchors: list, 元素为int锚点(一系列纵坐标)
    n_lanes: 指定的车道线条数
    
    RETURN
    coords_list: 车道线坐标，嵌套list, [Lane1[(x1,y1),(x2,y2),...], Lane2[], ...]
    coords_array: 车道线坐标，numpy.array, shape(n_lanes, len(anchors), 2)
    不管是list还是array，第一维(层)是车道，第二维(层)是各车道的坐标点，第三维(层)是每个坐标点的xy坐标
    '''   
    # n_lanes = n_lanes if n_lanes else np.max(seg) #TODO
    
    coords_list = [[] for _ in range(n_lanes)]
    coords_array = -np.ones((n_lanes,len(anchors),2)).astype(int)
    coords_array[:,:,1] = anchors
    
    for k,y in enumerate(anchors): #对每个anchor（纵坐标）
        #读取分割标签该anchor的数据
        label_row = np.asarray(seg)[int(round(y))] 
        for i in range(n_lanes): #对每条车道线
            #获取每个车道线该anchor（纵坐标）下的横坐标    
            x = np.where(label_row == i+1)[0] #分割标签车道线从1开始
                        
            if len(x) == 0: #如果该anchor行数据没有对应车道线的值则不添加
                continue    
            x = round(np.mean(x)) #如果有横坐标，取均值（车道线有宽度，分割结果在anchor行可能有多个相邻的像素都是对应车道线的值）
            coords_list[i].append((x,y)) #横坐标记均值，纵坐标记该anchor
            coords_array[i,k,0] = x
            coords_array[i,k,1] = y
    
    return coords_list, coords_array


def coord2seg(coords, seg_size, line_width = 15, instance = True): 
    '''
    根据坐标生成分割标签（或者分割结果）
    coords: 车道线坐标，嵌套list，[lane1[(x1,y1),...], lane2[(x1,y1),...], ...]
            或numpy.array, shape(n_lanes, len(anchors), 2)
    不管是list还是array，第一维(层)是车道，第二维(层)是各车道的坐标点，第三维(层)是每个坐标点的xy坐标
    seg_size: 要生成的分割图的尺寸shape(h,w)
    line_width: 将坐标连线时线的粗细
    instance: 生成分割图是实例分割（区分不同车道线，值为1,2,...)还是语义分割（各车道线均为1）

    RETURN
    seg: numpy.array, shape(h,w) 坐标对应的分割图
    '''
    seg = np.zeros(seg_size)
    
    for index, lane in enumerate(coords): #每条车道线
        # print(index)
        for i in range(len(lane)-1): #该车道线的点两两连线，i个点 i-1条线
            if lane[i][0] >=0 and lane[i+1][0] >=0:#array格式下，不存在的点横坐标<0，只挑选横坐标>0的点连接
                cv2.line(seg, 
                         lane[i],lane[i+1],
                         index+1 if instance else 1,#语义分割每个车道线值均为1。实例分割，不同车道线值不同，依次为1,2,...
                         thickness=line_width) 
    return seg
    


def coords_fit(coords,order=2):
    '''
    利用2次多项式拟合修正坐标，一般用于模型分割结果的修正
    coords: 车道线坐标，嵌套list，[lane1[(x1,y1),...], lane2[(x1,y1),...], ...]
            或numpy.array, shape(n_lanes, len(anchors), 2)
    order: 进行多项式拟合的多项式次数
    
    RETURN
    fit_coords: 多项式拟合修正后的坐标，嵌套List或numpy.array，类型与格式同coords
    '''
    # n_lanes = len(coords)
    #返回的数据类型与输入相同
    return_list = (type(coords)==list)

    if return_list:
        fit_coords = []
        for lane in coords:
            temp = []  
            if len(lane)!= 0: #该车道线有坐标时
                xs = [xy[0] for xy in lane]
                ys = [xy[1] for xy in lane]
                #用固定的纵坐标取推算此处横坐标，因此y是自变量
                p = np.polyfit(ys,xs,deg=order) #拟合参数
                f = np.poly1d(p) #拟合函数
                fit_xs = f(ys) #推算横坐标
                for x,y in zip(fit_xs,ys):
                    temp.append((round(x),y))
            fit_coords.append(temp)
    else:
        fit_coords = -np.ones_like(coords)
        fit_coords[:,:,1] = coords[:,:,1] #纵坐标（锚点）不变
        for i,lane in enumerate(coords): #array格式必定有坐标，但只保留x>0的坐标点
            xs = [xy[0] for xy in lane if xy[0]>0]
            ys = [xy[1] for xy in lane if xy[0]>0]
            if len(xs) >0:
                p = np.polyfit(ys,xs,deg=order) #拟合参数
                f = np.poly1d(p) #拟合函数
                fit_xs = f(ys) #推算横坐标
                for j,x,y in zip(range(len(ys)),fit_xs,ys):
                    fit_coords[i,j,0] = round(x)
                    fit_coords[i,j,1] = y
    return fit_coords        

    
if __name__ == '__main__':
    
    from PIL import Image


    data_path = r"examples/CULane_demo" #示例数据（完整数据集的一部分）
    
    
    ##读取训练文件列表
    train_list = data_path + r"/list/train_gt.txt"
    with open(train_list, 'r') as f:
        train_file = f.readlines()

    n_samples = len(train_file) #总数据条数
    samples = train_file[np.random.randint(n_samples)] #任选一条数据内容
      
    ##将内容按空格划分，依次为图片位置，分割标签位置，4条车道线存在情况
    img_name = samples.split(' ')[0]#图片位置 
    print('image file: ', img_name)  
    seg_name = samples.split(' ')[1]#分割标签位置
    print('segmentation file: ', seg_name) 
    lane_exist = samples.split(' ')[2:] #4条车道线存在情况
    lane_exist = [int(x) for x in lane_exist] #str转int
    print('lanes existance: ', lane_exist) 
    
    
    #读取图片和分割标签
    img = Image.open(data_path + img_name)
    img = np.array(img)
    seg = Image.open(data_path + seg_name)
    seg = np.array(seg)

    #展示图片和分割标签
    plt.imshow(img)
    plt.title('img')
    plt.show()
    print('image shape:', img.shape) 
    plt.matshow(seg)
    plt.title('seg')
    plt.colorbar()
    plt.show()
    print('segmentation shape:', seg.shape)
    
    
    
    n_lanes = 4 #CULane数据集有4条车道线
    #对四种车道线分别上红绿蓝黄四色
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)] 
    
    #根据分割绘制车道线
    vis = img.copy()
    for i in range(n_lanes):#依次在原图上绘制车道线
        vis[seg==i+1,:] = color[i]#分割结果中为车道线的像素点，在原图上进行重新绘制为指定颜色
    plt.imshow(vis)
    plt.title('img & seg')
    plt.show()
    
    
    #根据CULane数据集中，与图片同名文本文件，绘制坐标
    txt_name = img_name.replace('jpg','lines.txt')#同名文本文件位置 
    #打开文本文件并处理
    with open(data_path + txt_name, 'r') as f:
        annos = f.readlines()        
    #文本文件内容中每行代表一个车道，每个车道横纵坐标值交替排列，空格隔开
    annos = [lane.strip().split(' ') for lane in annos] #去除每行末的换行符再按空格分割
    annos = [[float(x) for x in lane]for lane in annos] #将所有元素从str变为float
    #创建坐标点
    coords = [[] for _ in range(n_lanes)] #创建一个嵌套列表，将标记点的横纵坐标作为一组存入
    for i in range(n_lanes): #对每条车道线
        if lane_exist[i] == 1: #该条车道线存在时
            lane_data = annos.pop(0) #取出1条数据
            for x,y in zip(lane_data[::2],lane_data[1::2]):#x,y是第i条车道线的横纵坐标
                coords[i].append((round(x),round(y))) #像素点坐标为整数，对xy进行四舍五入
    #对坐标点作图  
    vis = img.copy()
    for i,lane in enumerate(coords):#对每条车道线
        for xy in lane:#对车道线内的每个标记点
            cv2.circle(vis,xy,5,color[i],thickness=-1) #作图
    plt.imshow(vis)
    plt.title('img & coord')
    plt.show()
            

    #使用coord_fit函数修正坐标
    coords_ = coords_fit(coords)
    #对坐标点作图  
    vis = img.copy()
    for i,lane in enumerate(coords_):#对每条车道线
        for xy in lane:#对车道线内的每个标记点
            cv2.circle(vis,xy,5,color[i],thickness=-1) #作图
    plt.imshow(vis)
    plt.title('img & coord(fitted)')
    plt.show()
                

    
    #h590尺寸数据的车道线锚点
    anchors = [round(i*590/288) for i in CULane_anchors] #CULane_anchors针对的是h288尺寸的数据，进行换算
    ##只记录每个车道线在图中上述纵坐标时的横坐标
    #调用seg2coord函数将分割变为坐标，并在图片上绘制
    coords_list, coords_array = seg2coord(seg, anchors) 
    #使用坐标点作图
    vis = img.copy()
    for i,lane in enumerate(coords_array):#对每条车道线 #此处coords_array,coords_list均可
        for xy in lane:#对车道线内的每个标记点
            if xy[0] > 0:#横坐标大于0时（array格式的坐标点，不存在该点时横坐标为负）
                cv2.circle(vis,xy,5,color[i],thickness=-1) #作图
    plt.imshow(vis)
    plt.title('img & coord(from seg)')
    plt.show()
    
    #调用coord2seg函数将坐标变为分割，并在图片上绘制
    seg_ = coord2seg(coords, img.shape[:2], line_width = 15, instance = True)
    #使用分割标签作图
    vis = img.copy()
    for i in range(n_lanes):#依次在原图上绘制车道线
        vis[seg_ == i+1,:] = color[i]#分割结果中为车道线的像素点，在原图上进行重新绘制为指定颜色
    plt.imshow(vis)
    plt.title('img & seg(from coord)')
    plt.show()
    
    
    
