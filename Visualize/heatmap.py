import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


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
    plt.title('conv2 Weights Heatmap')
    plt.xlabel('Filter')
    plt.ylabel('Channel')
    plt.show()