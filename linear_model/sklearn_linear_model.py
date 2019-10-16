import numpy as np
from sklearn.linear_model import LinearRegression

# 数据集准备sklearn至少是二维的
x = np.array([[2, 4], [4, 6], [5, 7], [6, 8], [8, 10]])
y = np.array([8, 16, 20, 24, 32])

# 引入LinearRegression 模型
model = LinearRegression()
# 训练模型
model.fit(x, y)
# 预测 15 与 y变量之间的关系 4-> 20,8->50 等 结果是 [87.14285714]
pred = model.predict([[3, 6]])
# 输出预测值
print(pred)
# 输出回归计算出的系数值
print(model.coef_)
# 输出回归计算出的截距值
print(model.intercept_)

"""
5.71428571429 1.42857142857
87.1428571429
"""
