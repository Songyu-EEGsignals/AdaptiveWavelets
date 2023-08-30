import TrainAWD1
from AWD_model.DWT2d import DWT2d
import pickle
import numpy as np
# with open(r'./data_path/train_sub01.pkl', 'rb') as f:
#     train_loader = pickle.load(f)
# res = train_loader
with open(r'./data_path/test_sub01.pkl', 'rb') as f:
    test_loader = pickle.load(f)

import Model_1
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def heatmap(weight, x_ticks, x_labels, y_ticks, y_labels):
    # 绘制热力图
    # 应用颜色映射的平滑函数
    norm = mcolors.PowerNorm(gamma=2)
    plt.imshow(np.abs(weight), cmap='hot', aspect='auto', interpolation='bilinear', norm=norm)
    # 设置横纵坐标刻度和标签
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    # 设置 y 轴标签名竖着显示
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation='vertical', ha='right')

    plt.colorbar()
    plt.title('Heatmap of wavelet coefficients')
    plt.xlabel('Frequency')
    plt.ylabel('Channel')
    plt.savefig('figures/' + "小波系数1&2_标签(0)" + '.pdf', bbox_inches='tight')
    plt.show()


device = "cpu"

wt = DWT2d(wave_row=TrainAWD1.wave_row, wave_column=TrainAWD1.wave_column, mode='periodization', J=TrainAWD1.J,
           init_factor=1, noise_factor=0.0).to(device)

Model = Model_1.AWD_Model_Init(wt=wt)
Model.load_state_dict(torch.load(r"D:\pytorch1.8.0-python3.9.13\AdaptiveWavelets\final_model\AWD_Model_88.2.pth"))
weights = Model.state_dict()

# 选择要获取特征图的层
target_layer_1 = Model.wt
feature_maps1 = np.zeros((1,22,94),dtype=float)
feature_maps2 = np.zeros((1,22,94),dtype=float)
feature_maps = []
count = 1
i = 0
for batch_idx, (data, y) in enumerate(test_loader):
    m = 0

    for k in y:
        m += 1
        print(m)
        print()
        if int(k) == 0:
            i += int(y.shape[0])

            data_res = data[m-1,:,:,:]

            data_res = torch.unsqueeze(data_res,dim=0)

            data_res = data_res.to(device)
            y = y.to(device)

            output1 = target_layer_1(data_res)[0]
            output2 = target_layer_1(data_res)[3]
            # 将特征图转为numpy数组
            feature_map1 = output1.detach().numpy()
            feature_map2 = output2.detach().numpy()


            # feature_map = feature_map.squeeze()  # (32,22,750)
            feature_maps1 += np.sum(feature_map1, axis=0)
            feature_maps2 += np.sum(feature_map2, axis=0)


    # feature_map1 = np.sum(feature_map1, axis=0) / 8.0

temp1 = []
temp2 = []

temp1 = feature_maps1.squeeze() / i
temp2 = feature_maps2.squeeze() / i
# 将数组保存到文件中
# np.savetxt('./result_weights/wt_feature_map1_4.txt', temp1)
# np.savetxt('./result_weights/wt_feature_map2_4.txt', temp2)


feature_map = np.concatenate((temp1,temp2),axis=1)



    # feature_map = feature_map[0][0]
    # print(feature_map.shape) #(22,750)

    # 设置横纵坐标范围
x_ticks = np.arange(0, 188)  # 横坐标刻度
y_ticks = np.arange(0, 22)  # 纵坐标刻度

x_labels = []  # 横坐标标签

j = 0
for i in range(0, 188):
    if i % 3 == 0:
        x_labels.append(str(0 + j)+'hz')
        j += 1
    else:
        x_labels.append(' ')


y_labels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']  # 纵坐标标签
heatmap(feature_map, x_ticks, x_labels, y_ticks, y_labels)


