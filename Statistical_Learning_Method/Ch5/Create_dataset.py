import numpy as np
import pandas as pd
    

def create_dataset(dataset_num=1):
    if dataset_num == 1:
        dataset_tmp = np.array([['青年', '否', '否', '一般', '否'],
                                ['青年', '否', '否', '好', '否'],
                                ['青年', '是', '否', '好', '是'],
                                ['青年', '是', '是', '一般', '是'],
                                ['青年', '否', '否', '一般', '否'],
                                ['中年', '否', '否', '一般', '否'],
                                ['中年', '否', '否', '好', '否'],
                                ['中年', '是', '是', '好', '是'],
                                ['中年', '否', '是', '非常好', '是'],
                                ['中年', '否', '是', '非常好', '是'],
                                ['老年', '否', '是', '非常好', '是'],
                                ['老年', '否', '是', '好', '是'],
                                ['老年', '是', '否', '好', '是'],
                                ['老年', '是', '否', '非常好', '是'],
                                ['老年', '否', '否', '一般', '否']])
        feature_tmp = np.array(['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
        dataset = pd.DataFrame(dataset_tmp, columns=feature_tmp)
    elif dataset_num == 2:
        dataset = pd.read_csv('watermelon.csv', encoding='gbk')
        dataset = dataset.drop(columns=['编号', '密度', '含糖率'])
    
    return dataset
