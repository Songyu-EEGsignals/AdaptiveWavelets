import numpy as np
from heatmap import heatmap
from awave.models.eegnet2 import EEGNet
import torch
from fft import fft_tran
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
import pickle
with open(r'D:\pycharm_project\adaptive-wavelets-master\MI_data_pkl\train_sub11.pkl', 'rb') as f:
    train_loader = pickle.load(f)
with open(r'D:\pycharm_project\adaptive-wavelets-master\MI_data_pkl\test_sub11.pkl', 'rb') as f:
    test_loader = pickle.load(f)

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

for batch_idx, (data, y) in enumerate(train_loader):
    data = data.to(device)
    y = y.to(device)
    # 前向传播，获取指定层的输出
    output = target_layer(data)
    # 将特征图转为numpy数组
    feature_map = output.cpu().detach().numpy()
    print(feature_map.shape)

    feature_map2 = feature_map[0][0]
    print(feature_map2.shape)


    # feature_map = feature_map[0][0]
    # print(feature_map.shape) #(22,750)




    # 生成时间序列数据
    t = np.linspace(0, 3, 750)  # 时间范围为0到1，共1000个时间点
    channels_name = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']  # 纵坐标标签

    # 创建图像和子图
    fig, ax = plt.subplots()
    for i in range(0,22):
        signal = feature_map2[i][:750]
        l1, l2 = fft_tran(t, signal)
        # 绘制曲线
        ax.plot(l1[l1 > 0], l2[l1 > 0] / 2, '-', lw=2, label=channels_name[i])

    # 添加图例
    ax.legend()

    # plt.plot(l1[l1 > 0], l2[l1 > 0] / 2, '-', lw=2)
    plt.title('SeparableConv2d Feature Map FFT')
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('F(f)')
    plt.show()









