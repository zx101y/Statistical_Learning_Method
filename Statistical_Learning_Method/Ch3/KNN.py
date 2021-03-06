import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


class KNN(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, sample, k):
        dist = np.linalg.norm(self.X - sample, axis=1)  # 新样本与每个训练样本的距离
        idx = np.argsort(dist)  # 距离从小到大排序，取得对应的索引
        idx = idx[:min(len(idx), k)]  # 考虑前k近的训练样本（k大于训练样本数时考虑全部）
        
        label_count = {}  # 计算这k个训练样本中不同标签的数目
        for label in self.y[idx]:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        return max(label_count, key=label_count.get)  # 找出出现次数最多的标签，作为预测值


def read_dataset():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    # 把特征和标签连接起来
    dataset = np.concatenate((data[:, :2], target.reshape(-1, 1)), axis=1)
    
    return np.concatenate((dataset[0:40], dataset[50:90], dataset[100:140]), axis=0), np.concatenate((dataset[40:50], dataset[90:100], dataset[140:150]), axis=0)


if __name__ == '__main__':
    # 读取数据
    train_data, test_data = read_dataset()
    X_train, X_test, y_train, y_test = train_data[:, :-1], test_data[:, :-1], train_data[:, -1], test_data[:, -1]
    
    # 对测试集进行预测
    knn = KNN(X_train, y_train)
    k = 10
    y_test_pre = np.array([])
    for sample in X_test:
        y_test_pre = np.append(y_test_pre, knn.predict(sample, k))
    
    # 计算准确率
    correct_rate = sum(y_test_pre == y_test) / len(y_test)
    
    # Ground truth可视化
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='r')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='g')
    plt.scatter(X_train[y_train==2, 0], X_train[y_train==2, 1], color='b')
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], color='r', marker='*')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], color='g', marker='*')
    plt.scatter(X_test[y_test==2, 0], X_test[y_test==2, 1], color='b', marker='*')
    plt.show()

    # 预测可视化
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color='r')
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color='g')
    plt.scatter(X_train[y_train==2, 0], X_train[y_train==2, 1], color='b')
    plt.scatter(X_test[y_test_pre==0, 0], X_test[y_test_pre==0, 1], color='r', marker='*')
    plt.scatter(X_test[y_test_pre==1, 0], X_test[y_test_pre==1, 1], color='g', marker='*')
    plt.scatter(X_test[y_test_pre==2, 0], X_test[y_test_pre==2, 1], color='b', marker='*')
    plt.show()
