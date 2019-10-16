import numpy as np
from sklearn.linear_model import LinearRegression

# y = theta0+theta1*x1
x = np.array([[2, 4], [4, 6], [5, 7], [6, 8], [8, 10]])
y = np.array([8, 16, 20, 24, 32])
# 生成五个1
# x1 = np.ones((5, 1))
# 将生成的5个1与x组合
# x = np.c_[x1, x]
model = LinearRegression(fit_intercept=False)
# 使用说明
# LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
# fit_intercept:是否有截据，如果没有则直线过原点。
# normalize:是否将数据归一化。
# copy_X:默认为True，当为True时，X会被copied,否则X将会被覆写。（这一参数的具体作用没明白，求大神指教了）
# n_jobs:默认值为1。计算时使用的核数。
model.fit(x, y)
pred = model.predict([[3, 6]])
# [[ 89.7260274]]
# 输出预测值
print(pred)
# 输出回归计算出的系数值
print(model.coef_)
# 输出回归计算出的截距值
print(model.intercept_)

# 线性回归算法：y^{(i)}=\theta x^{(i)} + \varepsilon^{(i)}
#
#     我们要找到一种算法来求\theta，使模型预测效果最好。
#     假设误差\varepsilon属于高斯分布，得出误差\varepsilon的概率密度，
#     公式变换，得到实际值与预测值之间的关系
#     引入似然函数，即让似然函数的值越大越好
#     为了方便计算，左右两边取对数
#     得到线性回归算法的损失函数
#     针对损失函数优化\theta值
