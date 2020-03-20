import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
            'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])  # 截取前100个样本，提取前两个feature和label
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    
    return data[:, :2], data[:, -1]  # 分别返回feature和label


class SVM(object):
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel  # 核函数类型
    
    def init_args(self, features, labels):
        self.m, self.n = features.shape  # m是样本数，n是特征数
        self.X = features
        self.Y = labels
        self.b = 0.0
        
        self.alpha = np.ones(self.m)  # 拉格朗日方程里原约束条件的系数
        self.E = [self._E(i) for i in range(self.m)]  # 当前某个样本标签预测值和真实值的差
        self.C = 10.0  # 松弛变量的惩罚系数
    
    def _KKT(self, i):  # P147，判断KKT条件
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:  # 样本点在间隔边界内
            return (y_g >= 1)
        elif 0 < self.alpha[i] < self.C:  # 样本点在间隔边界上
            return (y_g == 1)
        else:  # self.alpha[i] == C，样本点在间隔边界外
            return (y_g <= 1)
    
    def _g(self, i):  # 计算第i个样本的预测值
        result = self.b
        for j in range(self.m):
            result += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        
        return result
    
    def kernel(self, x1, x2):
        if self._kernel == 'linear':  # 相当于不使用核函数，直接算内积
            return sum([x1[k] *x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':  # 多项式核函数
            return (sum([x1[k] *x2[k] for k in range(self.n)]) + 1) ** 2
        
        raise RuntimeError('Unexpected kernel type!')  # 未指定的核函数
    
    def _E(self, i):  # 计算第i个样本预测值和真实值（Y）的差
        return self._g(i) - self.Y[i]
    
    def _init_alpha(self):  # 找出下一步要更新alpha的两个样本点
        # 优先找满足0 < alpha < C的样本点
        index_list = [i for i in range (self.m) if 0 < self.alpha[i] < self.C]
        # 其次找不满足这一条件的样本点
        index_list2 = [i for i in range(self.m) if i not in index_list]
        index_list.extend(index_list2)
        
        for i in index_list:
            if self._KKT(i):  # 满足KKT条件的变量点的alpha不用改
                continue
            
            E1 = self.E[i]
            # 选择一个E2的下标，使得E1和E2的差的绝对值最大
            j = max(range(self.m), key=lambda x: abs(self.E[x] - E1))
            
            return i, j
    
    def _compare(self, _alpha, L, H):  # alpha_2的最优解限制在L和H之间
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha
    
    def fit(self, features, labels):
        self.init_args(features, labels)
        
        for t in range(self.max_iter):
            # 找出下一步要更新alpha的两个样本点
            i1, i2 = self._init_alpha()
            alpha1 = self.alpha[i1]
            alpha2 = self.alpha[i2]
            
            # 计算alpha_2的上下边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, alpha1 + alpha2 - self.C)
                H = min(self.C, alpha1 + alpha2)
            else:  # self.Y[i1] != self.Y[i2]
                L = min(0, alpha2 - alpha1)
                H = max(self.C, self.C + alpha2 - alpha1)
            
            # 取出E1、E2并计算eta（η）
            E1 = self.E[i1]
            E2 = self.E[i2]
            K11 = self.kernel(X[i1], X[i1])
            K22 = self.kernel(X[i2], X[i2])
            K12 = self.kernel(X[i1], X[i2])
            eta = K11 + K22 - 2 * K12
            if eta == 0:
                continue
            
            # 更新alpha
            alpha2_new_unc = alpha2 + self.Y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_new_unc, L, H)
            alpha1_new = alpha1 + self.Y[i1] * self.Y[i2] * (alpha2 - alpha2_new)
            
            # 更新b
            b1_new = -E1 - self.Y[i1] * K11 * (alpha1_new - alpha1) - self.Y[i2] * K12 * (alpha2_new - alpha2) + self.b
            b2_new = -E2 - self.Y[i1] * K12 * (alpha1_new - alpha1) - self.Y[i2] * K22 * (alpha2_new - alpha2) + self.b
            if 0 < alpha1_new < self.C and 0 < alpha2_new < self.C:  # 此时b1_new == b2_new
                b_new = b1_new
            else:
                b_new = (b1_new + b2_new) / 2
            
            # 写入更新值
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new
            
            # 重新计算E1和E2
            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)
    
    def predict(self, sample):
        result = self.b
        for i in range(self.m):
            result += self.alpha[i] * self.Y[i] * self.kernel(self.X[i], sample)
        
        return 1 if result > 0 else -1
    
    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        
        return right_count / len(X_test)
    
    def weight(self):
        x_mul_y = self.X * self.Y.reshape(-1, 1)
        w = np.dot(self.alpha.reshape(1, -1), x_mul_y)
        
        return w.reshape(-1)
        

if __name__ == '__main__':
    # 创造数据
    X, y = create_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # 可视化
    plt.scatter(X[:50, 0], X[:50, 1], label='0')  # 前50个是反例
    plt.scatter(X[50:, 0], X[50:, 1], label='1')  # 后50个是正例
    plt.legend()
    
    # 训练和测试
    svm = SVM(max_iter=1000)
    svm.fit(X_train, y_train)
    print(svm.score(X_test, y_test))
    
    # 调包试一下
    clf = SVC()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    