# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


'''
注意该模型把图片尺寸（宽高）压缩至原来的1/32后再恢复到原尺寸
如果图片尺寸不是32的整数倍，压缩时有截断，会导致恢复的尺寸与原尺寸不同
'''

#定义模型
class CNN(nn.Module):
    def __init__(self, n_cls = 4):
        #n_cls为车道线类数，每个像素共n_cls+1种可能（n_cls种车道和无车道）
        super().__init__()
        
        #构造encoder，有多层卷积，每一层卷积使数据尺寸（宽高）减半
        self.encoder1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.encoder3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.encoder4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.encoder5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        #构造decoder，有多层反卷积，每一层卷积使数据尺寸（宽高）加倍
        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(128, n_cls+1, kernel_size=4, stride=2, padding=1)   
        
    def forward(self,x): 
        #串行通过所有层，每层通过后使用ReLU激活
        #x:(bs,3,h,w)
        x = F.relu(self.encoder1(x)) #->(bs,16,h/2,w/2)
        x = F.relu(self.encoder2(x)) #->(bs,32,h/4,w/4)
        x = F.relu(self.encoder3(x)) #->(bs,64,h/8,w/8)
        x = F.relu(self.encoder4(x)) #->(bs,128,h/16,w/16)
        x = F.relu(self.encoder5(x)) #->(bs,256,h/32,w/32)
        
        x = F.relu(self.decoder1(x)) #->(bs,128,h/16,w/16)
        x = F.relu(self.decoder2(x)) #->(bs,n_cls+1=5,h/8,w/8)
            
        #将结果上采样至数据尺寸8倍    
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        #->(bs,n_cls+1=5,h,w)
        
        return x


if __name__ == '__main__':
    model = CNN()
    print(model) #模型结构
    #=> CNN(
    #=>   (encoder1): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=>   (encoder2): Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=>   (encoder3): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=>   (encoder4): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=>   (encoder5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=>   (decoder1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=>   (decoder2): ConvTranspose2d(128, 5, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #=> )
    
    dummy_img = torch.zeros(1,3,288,800) #创造一个输入数据
    pred = model(dummy_img) 
    print(dummy_img.shape, pred.shape) #模型输入、输出数据的尺寸
    #=> torch.Size([1, 3, 288, 800]) torch.Size([1, 5, 288, 800])