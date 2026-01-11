import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.gridspec as gridspec
# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = np.load('data/Node-independent_temporal_graph.npy')
i = 22
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
cmap = plt.get_cmap('YlOrRd')
titles = ['The heatmap of temporal \ngraph at 8:00',
          'The heatmap of temporal \ngraph at 12:00',
          'The heatmap of temporal \ngraph at 16:00',
          'The heatmap of temporal \ngraph at 20:00']
for ax, data, title in zip(axes, [data[i], data[i], data[i], data[i]], titles):
    im = ax.imshow(data, cmap=cmap)
    ax.set_xticks(np.arange(0, 13, 2))
    ax.set_yticks(np.arange(0, 13, 2))
    ax.set_title("", pad=0)
    ax.text(0, -0.3, title, transform=ax.transAxes, ha='left')
    ax.invert_yaxis()
cbar_ax = fig.add_axes([0.92, 0.24, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax)
plt.show()