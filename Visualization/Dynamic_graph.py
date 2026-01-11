import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = np.load('data/Dynamic graph weights.npy')
i = 50
j = 61
data = data[:, i, j]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(len(data)), data, width=0.8,color='royalblue', alpha=0.6)
new_ticks_pos = np.arange(0, 210, 50)
new_ticks_labels = [f"{i:02d}:00" for i in range(4, 24, 4)]
ax.set_xticks(new_ticks_pos)
ax.set_xticklabels(new_ticks_labels)
plt.xlabel('Time',fontsize=15)
plt.ylabel('Weight',fontsize=15)
plt.show()