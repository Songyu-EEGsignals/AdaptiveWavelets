'''
    EEGNet PyTorch implementation
    Original implementation - https://github.com/vlawhern/arl-eegmodels
    Original paper: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    ---
    EEGNet Parameters:

      nb_classes      : int, number of classes to classify
      Chans           : number of channels in the EEG data
      Samples         : sample frequency (Hz) in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer.
                        ARL recommends to set this parameter to be half of the sampling rate.
                        For the SMR dataset in particular since the data was high-passed at 4Hz ARL used a kernel length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
# from SeparableConv import SeparableConv1d

import torch.optim as optim


class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        stride=1,padding=self.padding, groups=self.c_in, bias=False)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=1, stride=1,bias=False)

        # 创建可训练参数并注册为模型参数
        # self.weight = nn.Parameter(torch.randn(c_out, c_in, 1, 15))


    def forward(self, x: torch.Tensor):
        # 对深度可分离卷积中的核函数进行约束
        # self.weight.data = torch.clamp(self.weight.data, min=0.0, max=1.0)

        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class DenseWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1,**kwargs):
        self.max_norm = max_norm
        super(DenseWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DenseWithConstraint, self).forward(x)

class EEGNet(nn.Module):

    def __init__(self, nb_classes: int=5, Chans: int = 22,
                 dropoutRate: float = 0.5, kernLength: int = 125,
                 F1:int = 16, D:int = 4):
        super().__init__()

        F2 = F1 * 2

        # Make kernel size and odd number
        try:
            assert kernLength % 2 != 0
        except AssertionError:
            raise ValueError("ERROR: kernLength must be odd number")

        # In: (B, Chans, Samples, 1)
        # Out: (B, F1, Samples, 1)
        self.conv1 = nn.Conv2d(1, F1, (1,kernLength), stride=1,padding=(0,62), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, eps=1e-3,  affine=True,momentum=0.99) # (B, F1, Samples, 1)
        # In: (B, F1, Samples, 1)
        # Out: (B, F2, Samples - Chans + 1, 1)
        self.conv2 = Conv2dWithConstraint(F1, F1*D, (Chans,1), groups=F1,max_norm=1, stride=1,padding=(0, 0),bias=False)


        self.bn2 = nn.BatchNorm2d(F1*D,eps=1e-03,  affine=True,momentum=0.99) # (B, F2, Samples - Chans + 1, 1)
        # In: (B, F2, Samples - Chans + 1, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.avg_pool = nn.AvgPool2d((1,4))
        self.dropout = nn.Dropout(p=dropoutRate)

        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 4, 1)
        self.conv3 = SeparableConv2d(64, F2, kernel_size=(1,16), padding=(0,8))
        self.bn3 = nn.BatchNorm2d(F2,eps=1e-03,  affine=True,momentum=0.99)
        # In: (B, F2, (Samples - Chans + 1) / 4, 1)
        # Out: (B, F2, (Samples - Chans + 1) / 32, 1)
        self.avg_pool2 = nn.AvgPool2d((1,8))
        # In: (B, F2 *  (Samples - Chans + 1) / 32)
        self.fc = DenseWithConstraint(736, nb_classes,max_norm=0.25)



        # self.norm = nn.utils.weight_norm(nn.Conv2d(1, F1, (1, kernLength), padding=(0, int((kernLength - 1) / 2)), bias=False))
        # self.norm1 = nn.utils.weight_norm(nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False))
        # self.norm2 = nn.utils.weight_norm(nn.Conv2d(F1 * D, F2, (1, 16), stride=(1, 1), padding=(0, 7), bias=False))


    def forward(self, x: torch.Tensor):
        # Block 1
        y1 = self.conv1(x)
        # print("conv1: ", y1.shape)
        y1 = self.bn1(y1)
        # print("bn1: ", y1.shape)
        y1 = self.conv2(y1)
        # print("conv2", y1.shape)
        y1 = F.elu(self.bn2(y1))
        # print("bn2", y1.shape)
        y1 = self.avg_pool(y1)
        # print("avg_pool", y1.shape)
        y1 = self.dropout(y1)
        # print("dropout", y1.shape)

        # Block 2
        y2 = self.conv3(y1)
        # print("conv3", y2.shape)
        y2 = F.elu(self.bn3(y2))
        # print("bn3", y2.shape)
        y2 = self.avg_pool2(y2)
        # print("avg_pool2", y2.shape)
        y2 = self.dropout(y2)
        # print("dropout", y2.shape)
        y2 = torch.flatten(y2, 1)
        # print("flatten", y2.shape)
        y2 = self.fc(y2)
        # print("fc", y2.shape)
        # y2 = self.fc1(y2)

        # print("fc", y2.shape)
        y2 = F.log_softmax(y2, dim=1)

        return y2