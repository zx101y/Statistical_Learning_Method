import numpy as np
from Data_ch4 import read_dataset


class NaiveBayes(object):
    def __init__(self):
        self.prior_prob = {}
        self.cond_prob = []
    
    def train(self, train_data, lambd=1):
        # lambd=0时，极大似然估计
        # lambd=1时，拉普拉斯平滑
        
        N = train_data.shape[0]
        y_vals = set(train_data[:, -1])
        K = len(y_vals)
        
        # 计算先验概率
        for y_val in y_vals:
            self.prior_prob[y_val] = (sum(train_data[:, -1] == y_val) + lambd) / (N + K * lambd)
        
        # 计算各个特征的条件概率
        for x_idx in range(train_data.shape[1]-1):
            self.cond_prob.append({})
            x_vals = set(train_data[:, x_idx])
            S = len(x_vals)
            
            for x_val in x_vals:
                for y_val in set(train_data[:, -1]):
                    self.cond_prob[-1][(x_val, y_val)] = (sum((train_data[:, x_idx] == x_val) & (train_data[:, -1] == y_val)) + lambd) / (sum(train_data[:, -1] == y_val) + S * lambd)
    
    def predict(self, data):
        prob = {}
        
        for y_val, y_prob in self.prior_prob.items():
            prob_temp = y_prob
            for x_idx in range(data.shape[0]):
                prob_temp *= self.cond_prob[x_idx][(data[x_idx], y_val)]
            prob[y_val] = prob_temp
        
        prob_reverse = {v:k for k, v in prob.items()}
        label = prob_reverse[max(prob.values())]
        
        return label, prob


if __name__ == '__main__':
    train_data = read_dataset()
    naive_bayes = NaiveBayes()
    naive_bayes.train(train_data, lambd=0)
    label, prob = naive_bayes.predict(np.array([2, 'S']))
    
    print(label)
    print(prob)
