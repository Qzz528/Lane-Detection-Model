# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 08:40:37 2024

@author: Administrator
"""
def segmodel_predict(img, model, size, device):
    '''
    img: PIL图片, 来源于Image.open(),Image.fromarray()
    model: torch模型
    size: tuple, 模型输入数据尺寸(h,w), img将被变形为该尺寸
    device:计算的设备

    SHOW
    img图片并绘制模型判断的车道线
    RETURN
    pred: numpy.array, 尺寸(h,w), 数值0代表背景, 1,2,...代表不同的车道线
    '''    
    vis = np.array(img) #复制一份原始数据以作图
    vis = cv2.resize(vis,(size[1],size[0])) #尺寸变换
    
    #对图片进行与训练集同样的处理，作为模型输入
    img_trans = img_transform_generate(*size)
    img = img_trans(img).to(device)
    img = img[None,:] #(3,h,w)->(1,3,h,w)
    
    model = model.to(device)
    
    pred = model(img) #(1,3,h,w)=model=>(1,5,h,w),...
    pred = pred[0] if type(pred) == tuple else pred #如果模型输出有多个则第一个为分割结果
    pred = pred[0] #(1,5,h,w)->(5,h,w)

    #每个像素点处有5个值（四种车道和无车道），其中最大值对应的通道序号是该点的车道线类别
    pred = pred.detach().cpu().numpy()  
    pred = np.argmax(pred,axis=0) #(5,h,w)->(h,w)

    #作图
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)] #对四种车道线分别上红绿蓝黄四色
    for i,c in zip(range(1,5),color):#依次在原图上绘制车道线
        vis[pred==i,:] = c#分割结果中为车道线的像素点，在原图上进行重新绘制为指定颜色
    plt.imshow(vis)
    plt.show()
    
    return pred #返回分割结果

if __name__ == '__main__':
    print('segmodel_predict')
    import torch
    model.load_state_dict(torch.load("weights\FCN_small.pt"))
    from PIL import Image
    img = Image.open('samp_img.jpg')
    pred = segmodel_predict(img, model, (288,800), torch.device('cpu'))
    plt.matshow(pred)
    plt.show()
    print(pred.shape)