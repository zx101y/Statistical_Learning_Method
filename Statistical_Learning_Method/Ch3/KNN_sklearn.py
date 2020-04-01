import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


def read_dataset():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    # 把特征和标签连接起来
    dataset = np.concatenate((data[:, :2], target.reshape(-1, 1)), axis=1)
    
    return np.concatenate((dataset[0:40], dataset[50:90], dataset[100:140]), axis=0), np.concatenate((dataset[40:50], dataset[90:100], dataset[140:150]), axis=0)


if __name__ == '__main__':
    # 读取和预测
    train_data, test_data = read_dataset()
    test_data_pre = test_data.copy()
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(train_data[:, :-1], train_data[:, -1])
    test_data_pre[:, -1] = knn.predict(test_data_pre[:, :-1])
    
    # Ground truth可视化
    plt.scatter(train_data[train_data[:, -1]==0, 0], train_data[train_data[:, -1]==0, 1], color='r')
    plt.scatter(train_data[train_data[:, -1]==1, 0], train_data[train_data[:, -1]==1, 1], color='g')
    plt.scatter(train_data[train_data[:, -1]==2, 0], train_data[train_data[:, -1]==2, 1], color='b')
    plt.scatter(test_data[test_data[:, -1]==0, 0], test_data[test_data[:, -1]==0, 1], color='r', marker='*')
    plt.scatter(test_data[test_data[:, -1]==1, 0], test_data[test_data[:, -1]==1, 1], color='g', marker='*')
    plt.scatter(test_data[test_data[:, -1]==2, 0], test_data[test_data[:, -1]==2, 1], color='b', marker='*')
    plt.show()
    
    # 预测结果可视化
    plt.scatter(train_data[train_data[:, -1]==0, 0], train_data[train_data[:, -1]==0, 1], color='r')
    plt.scatter(train_data[train_data[:, -1]==1, 0], train_data[train_data[:, -1]==1, 1], color='g')
    plt.scatter(train_data[train_data[:, -1]==2, 0], train_data[train_data[:, -1]==2, 1], color='b')
    plt.scatter(test_data_pre[test_data_pre[:, -1]==0, 0], test_data_pre[test_data_pre[:, -1]==0, 1], color='r', marker='*')
    plt.scatter(test_data_pre[test_data_pre[:, -1]==1, 0], test_data_pre[test_data_pre[:, -1]==1, 1], color='g', marker='*')
    plt.scatter(test_data_pre[test_data_pre[:, -1]==2, 0], test_data_pre[test_data_pre[:, -1]==2, 1], color='b', marker='*')
    plt.show()
