import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = np.load('data/Predictive Traffic Visualization08.npy')
x = np.linspace(4, 20, 192, endpoint=False)
ground_truth = data[0]
prediction = data[1]
plt.figure(figsize=(9, 6))
plt.xlabel('Time',fontsize=28)
plt.ylabel('Traffic Flow',fontsize=28)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(np.arange(4, 21, 4), [f'{i}:00' for i in range(4, 21, 4)], fontsize=20)
plt.plot(x, ground_truth, markersize=6, linestyle='-', label='Gound Truth')
plt.plot(x, prediction, markersize=4, linestyle='-', label='Prediction', color='red')
plt.legend(loc='upper right',fontsize=13)
plt.show()