# 引入数据集，波士顿房价
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 数据集的准备
dataset = load_boston()
# print(dataset.data.shape)
X = dataset['data']
y = dataset['target']
# train_test_split 切分数据集
# train训练集，test测试集
# 参数解释：
# train_data：被划分的样本特征集-此处为X
# train_target：被划分的样本标签-此处为y
# test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
# random_state：是随机数的种子。
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
# 随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
# 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 构建模型 进行训练

model = LinearRegression()
# 训练模型
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 得到预测值
print(y_pred)
# 评估
# 预测平均值的基准性能的均方根误差（RMSE）是约 9.21 千美元。
print("MAE: %s" % mean_absolute_error(y_test, y_pred))
print(y_test)
