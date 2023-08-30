import numpy as np
from awave.models.eegnet2 import EEGNet
import torch
from fft import fft_tran
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
import pickle
with open(r'D:\pycharm_project\adaptive-wavelets-master\MI_data_pkl\train_sub11.pkl', 'rb') as f:
    train_loader = pickle.load(f)
with open(r'D:\pycharm_project\adaptive-wavelets-master\MI_data_pkl\test_sub11.pkl', 'rb') as f:
    test_loader = pickle.load(f)


def heatmap(weight, x_ticks, x_labels, y_ticks, y_labels):
    # 绘制热力图
    # 应用颜色映射的平滑函数
    norm = mcolors.PowerNorm(gamma=1.5)
    plt.imshow(np.abs(weight), cmap='hot', aspect='auto', interpolation='bilinear', norm=norm)
    # 设置横纵坐标刻度和标签
    plt.xticks(x_ticks)
    plt.yticks(y_ticks, y_labels)
    # 设置 y 轴标签名竖着显示
    # plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical', ha='right')

    plt.colorbar()
    plt.title('SeparableConv2d Weights Heatmap')
    plt.xlabel('Filter')
    plt.ylabel('Channel')
    plt.show()

eegnet = EEGNet(Chans=22).to(device)
eegnet.load_state_dict(torch.load('../src/MI/EEGNet.pth'))  # 加载保存的模型权重
weights = eegnet.state_dict()

conv1_weight = eegnet.state_dict()['conv1.weight'].cpu().detach().numpy()
conv3_depthwise_conv_weight = eegnet.state_dict()['conv3.depthwise_conv.weight']
conv3_conv2d_1x1_weight = eegnet.state_dict()['conv3.conv2d_1x1.weight']
conv4_weight = eegnet.state_dict()['conv4.weight']

print(conv1_weight.shape)
print(conv3_depthwise_conv_weight.shape)
print(conv3_conv2d_1x1_weight.shape)
print(conv4_weight.shape)

# 选择要获取特征图的层
target_layer = eegnet.conv3
feature_maps = []
count = 1
for batch_idx, (data, y) in enumerate(train_loader):
    data = data.to(device)
    y = y.to(device)
    # 前向传播，获取指定层的输出
    output = target_layer(data)
    # 将特征图转为numpy数组
    feature_map = output.cpu().detach().numpy()
    # feature_map = feature_map.squeeze()  # (32,22,750)
    print(feature_map.shape)
    feature_map2 = np.sum(feature_map, axis=0) / 32.0
    print(feature_map2.shape)

    for i in range(0,5):

        feature_map3 = feature_map2[i]


        # feature_map = feature_map[0][0]
        # print(feature_map.shape) #(22,750)


        # 设置横纵坐标范围
        x_ticks = np.arange(0, 751, 50)  # 横坐标刻度
        y_ticks = np.arange(0, 22)  # 纵坐标刻度


        x_labels = []  # 横坐标标签
        for i in range(0, 751):
            x_labels.append('time-' + str(i + 1))
        y_labels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                    'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']  # 纵坐标标签
        heatmap(feature_map3, x_ticks, x_labels, y_ticks, y_labels)
    exit(0)








# 转换权重维度
conv1_weight = conv1_weight.squeeze()
# 将权重数据调整为二维形状
conv1_weight = np.expand_dims(conv1_weight, axis=0)
print(conv1_weight)


# 绘制热力图
plt.imshow(conv1_weight, cmap='hot', aspect='auto')
plt.colorbar()
plt.title('Conv1 Weights Heatmap')
plt.xlabel('Weight Index')
plt.ylabel('Channel')
plt.show()



# weights = eegnet.state_dict()['/src/MI/EEGNET.weight']



