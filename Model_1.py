import torch.nn as nn
import torch.nn.functional as F
import torch


class SeparableConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: tuple, padding: tuple = 0):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.depthwise_conv = nn.Conv2d(self.c_in, self.c_in, kernel_size=self.kernel_size,
                                        stride=1,padding=self.padding, groups=self.c_in, bias=False)
        self.conv2d_1x1 = nn.Conv2d(self.c_in, self.c_out, kernel_size=(22,1), stride=1,bias=False)
    def forward(self, x: torch.Tensor):
        y = self.depthwise_conv(x)
        y = self.conv2d_1x1(y)
        return y
# class Conv2dWithConstraint(nn.Conv2d):
#     def __init__(self, *args, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         self.weight.data = torch.renorm(
#             self.weight.data, p=2, dim=0, maxnorm=self.max_norm
#         )
#         return super(Conv2dWithConstraint, self).forward(x)
class DenseWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0,**kwargs):
        self.max_norm = max_norm
        super(DenseWithConstraint, self).__init__(bias=False,*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DenseWithConstraint, self).forward(x)
class AWD_Model_Init(nn.Module):
    def __init__(self, wt):
        super(AWD_Model_Init, self).__init__()
        self.wt = wt.eval()
        self.wt.J = 3
        for param in wt.parameters():
            param.requires_grad = False
        self.Conv2_1 = SeparableConv2d(2,8,kernel_size=(1,5),padding=(0,1))
        self.Conv2_2 = SeparableConv2d(2,8,kernel_size=(1,3),padding=(0,0))
        self.Conv2_3 = SeparableConv2d(2,8,kernel_size=(1,7),padding=(0,2))
        self.bn3 = nn.BatchNorm2d(24,eps=1e-03,  affine=True,momentum=0.99)
        self.avg_pool = nn.AvgPool2d((1,4))
        self.fc1 = DenseWithConstraint(552, 16,max_norm=0.125)
        self.fc2 = DenseWithConstraint(16, 5,max_norm=0.25)
    def forward(self, x):
        batch_size = x.shape[0]
        x_t = self.wt(x)
        x_t0 = x_t[0]
        x_t3 = x_t[3]

        x_1 = torch.cat([x_t0, x_t3],dim=1)

        x_1_1 = self.Conv2_1(x_1)
        x_1_2 = self.Conv2_2(x_1)
        x_1_3 = self.Conv2_3(x_1)

        x = torch.cat([x_1_1,x_1_2,x_1_3],dim=1)

        x = F.elu(self.bn3(x))

        x = self.avg_pool(x)

        x = torch.flatten(x,1)

        x = self.fc1(x)
        x = self.fc2(x)
        self.wt.J = self.wt.J
        return x