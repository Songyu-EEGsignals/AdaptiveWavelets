import numpy as np

# from awave.models.eegnet2 import EEGNet
from awave.models.eegnet_pt import EEGNet
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


eegnet = EEGNet(Chans=22).to(device)
eegnet.load_state_dict(torch.load('../src/MI/EEGNet_89.9.pth'))  # 加载保存的模型权重
weights = eegnet.state_dict()
# print(weights)
# exit(0)

conv1_weight = eegnet.state_dict()['conv1.weight'].cpu().detach().numpy()
conv2_weight = eegnet.state_dict()['conv2.weight'].cpu().detach().numpy()
conv3_depthwise_conv_weight = eegnet.state_dict()['conv3.depthwise_conv.weight']
conv3_conv2d_1x1_weight = eegnet.state_dict()['conv3.conv2d_1x1.weight']
# conv4_weight = eegnet.state_dict()['conv4.weight']

print(conv1_weight.shape)
print(conv2_weight.shape)
print(conv3_depthwise_conv_weight.shape)
print(conv3_conv2d_1x1_weight.shape)
# print(conv4_weight.shape)

# 转换权重维度
conv2_weight = conv2_weight.squeeze()
# 将权重数据调整为二维形状
# conv1_weight = np.expand_dims(conv1_weight, axis=0)
# print(conv1_weight)

# 设置横纵坐标范围
x_ticks = np.arange(0, 64)  # 横坐标刻度
y_ticks = np.arange(0, 22)  # 纵坐标刻度

x_labels = []  # 横坐标标签
for i in range(0,64):
    x_labels.append('filter-'+str(i+1))
y_labels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']  # 纵坐标标签

conv2_weight = conv2_weight.reshape(22, 64)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 绘制热力图
# 应用颜色映射的平滑函数
norm = mcolors.PowerNorm(gamma=1)
plt.imshow(np.abs(conv2_weight), cmap='hot', aspect='auto', interpolation='bilinear', norm=norm)
# 设置横纵坐标刻度和标签
plt.xticks(x_ticks, x_labels)
plt.yticks(y_ticks, y_labels)
# 设置 y 轴标签名竖着显示
plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical', ha='right')


plt.colorbar()
plt.title('conv2 Weights Heatmap')
plt.xlabel('Filter')
plt.ylabel('Channel')
plt.show()



# weights = eegnet.state_dict()['/src/MI/EEGNET.weight']



