"""
knn
样本集
X = [[1,1],[1,1.5],[2,2],[4,3],[4,4]]
Y =['A','A','A','B','B']
测试样本:
t=[3,2] 属于哪个类型
设k=3
knn的实现：
    将待测试样本和所有的训练进行计算，找出最短的k个元素
    然后投票
"""

import operator

import numpy as np

X = np.array([[1, 1], [1, 1.5], [2, 2], [4, 3], [4, 4]])
y = np.array(['A', 'A', 'A', 'B', 'B'])


def knn_clasification(X, y, k, testSample):
    """
    :param X:   输入数据
    :param y:   真实数据
    :param k:   邻居的个数
    :param testSample:  待测试样本
    :return:    返回待测试样本类别
    """
    # 1.求待测试样本和所有训练样本之间的距离
    dist = np.sum((X - testSample) ** 2, axis=1) ** 0.5
    print('待测试样本和所有训练样本之间的距离是 %s' % dist)
    # 2.距离排序
    sort_dist = np.argsort(dist)
    print('距离排序后的结果(注意是下标) %s' % sort_dist)
    # 3.统计距离最近得k个样本的类别的个数
    class_count = {}
    for dist_index in sort_dist[:k]:
        label = y[dist_index]
        print('下标 %s ,内容是: %s' % (dist_index, label))
        class_count[label] = class_count.get(label, 0) + 1
        print('类型数量累加结果: %s ' % class_count)
    return sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)[0][0]


if __name__ == "__main__":
    # predict(预测,预知,预报)
    predict = knn_clasification(X, y, 3, np.array([3, 2]))
    print('预测的结果是: %s ' % predict)
