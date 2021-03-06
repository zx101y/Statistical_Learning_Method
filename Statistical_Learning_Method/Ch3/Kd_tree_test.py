import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Kd_tree import KdTree


def read_dataset(test_size=0.2):
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    # 把特征和标签连接起来
    dataset = np.concatenate((data[:, :2], target.reshape(-1, 1)), axis=1)
    
    n_test = int(50 * 0.2)
    train_data = np.concatenate((dataset[0:50-n_test], dataset[50:100-n_test], dataset[100:150-n_test]), axis=0)
    test_data = np.concatenate((dataset[50-n_test:50], dataset[100-n_test:100], dataset[150-n_test:150]), axis=0)
    
    return train_data, test_data

    
if __name__ == '__main__':
    # 读取数据
    train_data, test_data = read_dataset()
    test_data_pre = test_data.copy()
    
    # Ground truth可视化
    plt.scatter(train_data[train_data[:, -1]==0, 0], train_data[train_data[:, -1]==0, 1], color='r')
    plt.scatter(train_data[train_data[:, -1]==1, 0], train_data[train_data[:, -1]==1, 1], color='g')
    plt.scatter(train_data[train_data[:, -1]==2, 0], train_data[train_data[:, -1]==2, 1], color='b')
    plt.scatter(test_data[test_data[:, -1]==0, 0], test_data[test_data[:, -1]==0, 1], color='r', marker='*')
    plt.scatter(test_data[test_data[:, -1]==1, 0], test_data[test_data[:, -1]==1, 1], color='g', marker='*')
    plt.scatter(test_data[test_data[:, -1]==2, 0], test_data[test_data[:, -1]==2, 1], color='b', marker='*')
    plt.show()
    
    # 训练一棵Kd树，并用于预测
    kd_tree = KdTree(train_data)
    for i in range(len(test_data_pre)):
        test_data_pre[i, -1] = kd_tree.predict(test_data_pre[i, :-1], 10)
    
    # 预测结果可视化
    plt.scatter(train_data[train_data[:, -1]==0, 0], train_data[train_data[:, -1]==0, 1], color='r')
    plt.scatter(train_data[train_data[:, -1]==1, 0], train_data[train_data[:, -1]==1, 1], color='g')
    plt.scatter(train_data[train_data[:, -1]==2, 0], train_data[train_data[:, -1]==2, 1], color='b')
    plt.scatter(test_data_pre[test_data_pre[:, -1]==0, 0], test_data_pre[test_data_pre[:, -1]==0, 1], color='r', marker='*')
    plt.scatter(test_data_pre[test_data_pre[:, -1]==1, 0], test_data_pre[test_data_pre[:, -1]==1, 1], color='g', marker='*')
    plt.scatter(test_data_pre[test_data_pre[:, -1]==2, 0], test_data_pre[test_data_pre[:, -1]==2, 1], color='b', marker='*')
    plt.show()
    