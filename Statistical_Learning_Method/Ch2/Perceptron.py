from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self):
        self.w = np.random.randn(2)
        self.b = 0
    
    def cal_loss(self, features, labels):
        # 只考虑误分类项，即样本loss大于0的项
        return sum(np.maximum(-labels * (np.dot(features, self.w) + self.b), 0))
    
    def train(self, features, labels, learning_r, epsilon=0):
        loss = self.cal_loss(features, labels)
        while loss > epsilon:
            temp = labels * (np.dot(features, self.w) + self.b)
            idx = np.where(temp<=0)[0][0]  # 得到第一个分类错误样本的索引
            self.w += learning_r * labels[idx] * features[idx]
            self.b += learning_r * labels[idx]
            
            loss = self.cal_loss(features, labels)
    
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
    