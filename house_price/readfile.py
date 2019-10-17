import numpy as np

dataset = np.loadtxt('housing.data', dtype=np.object)
# 输出记录数列数
# (506, 14)
print(dataset.data.shape)
# print(dataset[:, :3])
# print((dataset[:, -1]))
# X = dataset[:,:3].astype(np.float)
# y = dataset[:,-1]
# print(y)
