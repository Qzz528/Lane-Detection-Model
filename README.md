LaneDetection-PyTorch-CULane

- utils（配套工具函数）
  - data_display：对数据进行作图展示
  - data_augment：对数据进行变换扩充
  - data_metric：对模型进行指标评价
- models（模型）

各种模型

- datasets（数据集，与模型配套使用）

- 与models配套，

如要修改对图片的处理方法 如加入normalize，请在datasets.py中修改

123

























FCN(FCN8s FCN16s FCN32s)

SegNet(SegNet,SegNet_VGG)

UNet(UNet,UNet_VGG)

ENet

PSPNet

CRFasRNN

DeepLab



Lanenet

SpatialCNN

UltraFastLaneDetection

## PolyLaneNet

GANet

LaneATT

lineCNN



ParseNet

RefineNet

ReSeg

LSTM-CF

DeepMask



Dilation conv  **Large Kernel Matters**

Inception Xception

PixelNet

Transformer

MobileNetV1/

SPP PPM FPN





FCN pretrainedBackbone, deconv, multioutput-add

UNet encoder-decoder，upsample，multioutput-concat, cba

SegNet unpool，

model = nn.Sequential(*list(resnet18(pretrained=False).children())[:-2])

替换encoder

特征提取模块分类网络



提取图像特征，根据特征还原到

FCN 借助vgg的features模块

SegNet

















FCN-8 16 32

FCN-vgg16-

FCN-Resnet19-



压缩数据 恢复数据

resnset也可以 bn效果更好

原文

espnet

pspnet resnet withoutbackground

lanenet without clustering( only featureing ,no lanes number limits)

hnet seg2coord

lanes function params

postprocess fit 





mean denoise
hist lightchange

img 光一致化 所有照片均调整亮度至



sequentail paralel modelstructure





argmax 时梯度无法反向传播 因此softmax
单调 判断时最大即可 不必softmax
loss 





sgd fit poly
backward 神经网络原理 每一层loss



chmod
iptable
nvidia-smi
killall

