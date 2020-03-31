from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self):
        self.alpha = None
        self.b = None
        self.w = None
    
    def train(self, features, labels, learning_r):
        self.alpha = np.zeros(len(features))
        self.b = 0
        
        while True:
            # 注意把alpha*y的结果转成一列
            self.w = ((self.alpha * labels).reshape(-1, 1) * features).sum(axis=0)
            temp = labels * (np.dot(features, self.w) + self.b)
            
            if len(np.where(temp<=0)[0]) == 0:  # 不存在误分类样本
                break
            else:
                idx = np.where(temp<=0)[0][0]  # 得到第一个误分类样本的索引
            
            self.alpha[idx] += learning_r
            self.b += learning_r * labels[idx]

    def model(self):
        return self.w, self.b
    
    def predict(self, feature):
        return np.sign(np.dot(feature, self.w) + self.b)


def read_dataset():
    iris = datasets.load_iris()
    data = iris.data
    feature_names = iris.feature_names
    target = iris.target
    
    dataset = pd.DataFrame(data, columns=feature_names).iloc[:, :2]  # 取前两个特征
    dataset['label'] = target  # 标签
    dataset.loc[dataset['label']==0, 'label'] = -1  # 标签0改为-1
    dataset = dataset.iloc[:100]  # 取前100个样本
    
    return dataset


if __name__ == '__main__':
    # 读取数据
    dataset = read_dataset()
    features = dataset.iloc[:, :2].values
    labels = dataset.iloc[:, 2].values
    
    # 创建并训练感知机
    perceptron = Perceptron()
    perceptron.train(features, labels, learning_r=1.2)
    w, b = perceptron.model()
    
    # 可视化
    h1 = plt.scatter(dataset.iloc[:50, 0], dataset.iloc[:50, 1], color='r')
    h2 = plt.scatter(dataset.iloc[50:, 0], dataset.iloc[50:, 1], color='g')
    plt.xlabel(dataset.columns[0])
    plt.ylabel(dataset.columns[1])
    plt.legend(handles=[h1, h2], labels=[0, 1])
    plt.plot([4.5, 7], [-w[0]/w[1]*4.5-b/w[1], -w[0]/w[1]*7-b/w[1]])
    plt.show()
    