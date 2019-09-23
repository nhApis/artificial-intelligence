import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1, 1], [1, 1.5], [2, 2], [4, 3], [4, 4]])
y = np.array(['A', 'A', 'A', 'B', 'B'])

# n_neighbors 就是K值
knn = KNeighborsClassifier(n_neighbors=3)
# 训练模型
knn.fit(X, y)

x_test = np.array([[3, 2],
                   [4, 5],
                   [10, 10]])

# 预测结果标签
pred = knn.predict(x_test)
# 预测样本为某个标签的概率,并且每一行的概率和为1
pred_proba = knn.predict_proba(x_test)
print(pred)
print(pred_proba)
