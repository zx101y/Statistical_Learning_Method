import numpy as np
import random


N = 10 #生成训练数据的个数

# AX=0 相当于matlab中 null(a','r')
def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()

# 符号函数，之后要进行向量化    
def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1
    
# noisy=False，那么就会生成N的dim维的线性可分数据X，标签为y
# noisy=True, 那么生成的数据是线性不可分的,标签为y      
def mk_data(N, noisy=False):
    rang = [-1,1]
    dim = 2
    
    X=np.random.rand(dim,N)*(rang[1]-rang[0])+rang[0]
    
    while True:
        Xsample = np.concatenate((np.ones((1,dim)), np.random.rand(dim,dim)*(rang[1]-rang[0])+rang[0]))
        k,w=null(Xsample.T)
        y = sign(np.dot(w.T,np.concatenate((np.ones((1,N)), X))))
        if np.all(y):
            break
            
    if noisy == True:
        idx = random.sample(range(1,N), N/10)
        y[idx] = -y[idx]
    
    return (X,y,w)
    

if __name__ == '__main__':
    sign = np.vectorize(sign)
    X,y,w = mk_data(10)
