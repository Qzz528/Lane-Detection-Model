# -*- coding: utf-8 -*-
import numpy as np
import torch

#从每批的分割标签和模型结果求取TP,FN,FP样本统计个数
def get_metric(pred, seg, n_cls=4, iou_thresh = 0.5):
    #pred 模型判断结果, seg 分割标签 形状均为(bs,h,w) numpyArray
    #其中每个像素值为0~n_cls的整数，表明该点所属n_cls+1种情况中的哪一种
    #0类是背景（无车道），1~n_cls是不同种车道线
    #返回该批次模型判断的TP,FN,FP
    
    TP,FN,FP = 0,0,0
    
    for i in range(n_cls+1):
        if i == 0: #无车道线不计算
            continue
        else:  #对每条车道线的该批样本            
            a = (pred==i) #模型判断pred中各个像素为第i条车道线的bool值
            b = (seg==i) #分割标签seg中各个像素为第i条车道线的bool值            
            union = a+b #bool值的和为并集 #一个为True，和为True
            inter = a*b #bool值的积为交集 #均为Ture，积为Ture

            n_a = np.sum(a,axis=(1,2)) #预测车道线像素数（为0表示不存在车道线）
            n_b = np.sum(b,axis=(1,2)) #标签车道线像素数（为0表示不存在车道线）
            n_u = np.sum(union,axis=(1,2)) #预测和标签并集像素数
            n_i = np.sum(inter,axis=(1,2)) #预测和标签交集像素数

            segP = (n_b!= 0) #分割标签中存在第i条车道线的正样本index
            iou = n_i[segP] / n_u[segP] #对标签实际存在车道线的样本计算预测和标签的交并比

            tp = np.sum(iou>=iou_thresh) #超过阈值，模型判断为正例（车道线）且正确判断
            fn = sum(segP)-tp 
            # fn = np.sum(iou<iou_thresh) #不超过阈值，标签存在车道但模型判断为非，漏检
            # assert tp+fn == sum(segP)

            
            predP = (n_a!= 0) #预测中存在第i条车道线的正样本index   
            fp = sum(predP)-tp
            # iou = n_i[predP] / n_u[predP] #对预测为车道线的样本计算预测和标签的交并比
            # fp = np.sum(iou<iou_thresh) #不超过阈值，模型判断为有车道线但判断错误，误检
            # assert tp+fp == sum(predP)

            #这种方法不计算标签无车道线且模型预测无车道线的情况(TN)
            TP += tp; FN += fn; FP += fp
    
    return TP,FN,FP


#从TP,FN,FP样本统计个数求取各种统计指标
def get_score(TP,FN,FP):
    eps=1e-8 #指标中分母加一个极小值避免分母为0
    per = TP/(TP+FP+eps) #查准率（精确率），所有模型正例判断中，确实是正样本的比例
    rec = TP/(TP+FN+eps) #查全率（召回率），所有样本正例中，模型判断出正例的比例
    f1 = (2*per*rec)/(per+rec+eps) #f1score 综合查准和查全
    return per,rec,f1


if __name__ == '__main__':
    #构造两个数据来测试
    pred = torch.zeros(3,4)
    pred[0,2]=pred[1,1]=pred[2,0]=1
    seg = torch.zeros(3,4)
    seg[:,1] = 1
    
    pred = pred[None,:].numpy() #(1,3,4)
    seg = seg[None,:].numpy() #(1,3,4)
    
    #数字为0说明无车道，为1说明是1号车道
    print(pred) 
    #=> [[[0. 0. 1. 0.]
    #=>   [0. 1. 0. 0.]
    #=>   [1. 0. 0. 0.]]]
    print(seg)
    #=> [[[0. 1. 0. 0.]
    #=>   [0. 1. 0. 0.]
    #=>   [0. 1. 0. 0.]]]
    pred = (pred==1)
    seg = (seg==1)
    print(np.sum(pred*seg)/np.sum(pred+seg)) #交并比
    #=> 0.2
    
    #单个车道线情况，阈值0.5时（交并比低于阈值）
    print(get_metric(pred, seg, 1, 0.5))
    #=> (0, 1, 1) #判断处无车道FP，实际车道处无判断FN
    
    #单个车道线情况，阈值0.1时（交并比高于阈值）
    print(get_metric(pred, seg, 1, 0.1))
    #=> (1, 0, 0) #预测车道和实际车道相符，TP
    