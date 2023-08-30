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
    plt.xlabel('Filter')
    plt.ylabel('Channel')
    plt.savefig('figures/' + "小波系数1&2_标签00" + '.pdf', bbox_inches='tight')
    plt.show()


device = "cpu"

wt = DWT2d(wave_row=TrainAWD1.wave_row, wave_column=TrainAWD1.wave_column, mode='periodization', J=TrainAWD1.J,
           init_factor=1, noise_factor=0.0).to(device)

Model = Model_1.AWD_Model_Init(wt=wt)
Model.load_state_dict(torch.load(r"D:\pytorch1.8.0-python3.9.13\AdaptiveWavelets\final_model\AWD_Model_88.2.pth"))
weights = Model.state_dict()

# 保存OrderedDict对象到文件
with open('model_weights.pkl', 'wb') as file:
    pickle.dump(weights, file)
