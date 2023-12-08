# -*- coding: utf-8 -*-
import numpy as np
import torch

#从每批的分割标签和模型结果求取TP,FN,TN,FP样本统计个数
def get_metric(pred, seg, n_cls=4, iou_thresh = 0.5):
    #pred 模型判断结果, seg 分割标签 形状均为(bs,h,w) numpyArray
    #其中每个像素值为0~n_cls的整数，表明该点所属n_cls+1种情况中的哪一种
    #0类是无车道背景，1~n_cls是不同种车道线
    #返回该批次模型判断的tp fp tn fn
    
    tp,fn,tn,fp = 0,0,0,0
    for i in range(n_cls+1):
        if i == 0: #无车道线不计算
            continue
        else:  #只计算车道线            
            a = (pred==i) #模型判断pred中各个像素为第i条车道线的bool值
            b = (seg==i) #分割标签seg中各个像素为第i条车道线的bool值            
            union = a+b #bool值的和为并集 #一个为True，和为True
            inter = a*b #bool值的积为交集 #均为Ture，积为Ture

            n_a = np.sum(a,axis=(1,2)) #预测车道线像素数（为0表示不存在车道线）
            n_b = np.sum(b,axis=(1,2)) #标签车道线像素数（为0表示不存在车道线）
            n_u = np.sum(union,axis=(1,2)) #预测和标签并集像素数
            n_i = np.sum(inter,axis=(1,2)) #预测和标签交集像素数

            segT = (n_b!= 0) #分割标签中存在第i条车道线的正样本index
            iou = n_i[segT] / n_u[segT] #对存在车道线的样本计算预测和标签的交并比

            tp += np.sum(iou>=iou_thresh) #超过阈值，模型判断为正例（车道线）且正确判断
            fn += np.sum(iou<iou_thresh) #不超过阈值，模型判断为非车道线但判断错误，漏检
            # assert TP+FN == sum(segT)

            segF = ~segT #分割标签中不存在第i条车道线的负样本index   
            tn += sum(n_a[segF] == 0) #预测结果内也无车道线，模型判断为负例且正确判断
            fp += sum(n_a[segF] != 0) #模型判断为正例（右车道线）但判断错误，误检
            # assert TN+FP == sum(segF)

    # assert TP+FN+TN+FP == pred.shape[0]*(n_cls-1)
    return tp,fn,tn,fp

#从TP,FN,TN,FP样本统计个数求取各种统计指标
def get_score(tp,fn,tn,fp):
    eps=1e-8 #指标中分母加一个极小值避免分母为0
    acc = (tp+tn)/(tp+fn+tn+fp) #准确率，所有正负样本中正确判断的比例
    per = tp/(tp+fp+eps) #查准率（精确率），所有模型正例判断中，确实是正样本的比例
    rec = tp/(tp+fn+eps) #查全率（召回率），所有样本正例中，模型判断出正例的比例
    f1 = (2*per*rec)/(per+rec+eps) #f1score 综合查准和查全
    return acc,per,rec,f1


if __name__ == '__main__':
    #构造两个数据来测试
    pred = torch.zeros(3,4)
    pred[0,2]=pred[1,1]=pred[2,0]=1
    seg = torch.zeros(3,4)
    seg[:,1] = 1
    
    pred = pred[None,:].numpy() #(1,3,4)
    seg = seg[None,:].numpy() #(1,3,4)
    
    print(pred)
    #=> [[[0. 0. 1. 0.]
    #=>   [0. 1. 0. 0.]
    #=>   [1. 0. 0. 0.]]]
    print(seg)
    #=> [[[0. 1. 0. 0.]
    #=>   [0. 1. 0. 0.]
    #=>   [0. 1. 0. 0.]]]
    
    print(get_metric(pred, seg, 1, 0.5))
    #=> (0, 1, 0, 0) #只有一条车道，iou阈值0.5时，认为预测车道和实际车道不符，FP
    print(get_metric(pred, seg, 1, 0.1))
    #=> (1, 0, 0, 0) #只有一条车道，iou阈值0.1时，认为预测车道和实际车道相符，TP
    print(get_metric(pred, seg, 2, 0.1))
    #=> (1, 0, 1, 0)
    #有两条车道，iou阈值0.1时，认为序号为1对应的车道预测和实际相符，TP
    #同时序号为2的车道预测和实际都不存在，TN