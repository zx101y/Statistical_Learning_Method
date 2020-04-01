import math
from Create_dataset import create_dataset


# 计算整个dataset的经验熵，dataset最后一列为标签
def cal_entropy(dataset):
    entropy = 0
    label_count = {}
    n = len(dataset)
    
    for i in range(n):
        data = dataset.iloc[i]
        if data[-1] in label_count:
            label_count[data[-1]] += 1
        else:
            label_count[data[-1]] = 1
        
    for label in label_count:
        p = label_count[label] / n
        entropy += (-p * math.log(p, 2))
    
    return entropy


# 计算dataset关于某个feature给定条件下的条件经验熵，axis指定feature所在列
def cal_cond_entropy(dataset, axis=0):
    cond_entropy = 0
    div_index = {}
    feat_count = {}
    n = len(dataset)
    
    for i in range(n):
        data = dataset.iloc[i]
        if data[axis] in div_index:
            div_index[data[axis]].append(i)
            feat_count[data[axis]] += 1
        else:
            div_index[data[axis]] = [i]
            feat_count[data[axis]] = 1
    
    for feat_val in div_index:
        fre = feat_count[feat_val] / n
        cond_entropy += fre * cal_entropy(dataset.iloc[div_index[feat_val]])
    
    return cond_entropy


# 计算dataset关于某个feature取值的熵，axis指定feature所在列
def cal_feat_entropy(dataset, axis=0):
    feat_entropy = 0
    feat_count = {}
    n = len(dataset)
    
    for i in range(n):
        data = dataset.iloc[i]
        if data[axis] in feat_count:
            feat_count[data[axis]] += 1
        else:
            feat_count[data[axis]] = 1
    
    for feat_val in feat_count:
        p = feat_count[feat_val] / n
        feat_entropy += (-p * math.log(p, 2))
        
    return feat_entropy


# 计算给定feature对dataset的信息增益，axis指定feature所在列
def cal_info_gain(dataset, axis=0):
    return cal_entropy(dataset) - cal_cond_entropy(dataset, axis)


# 计算给定feature对dataset的信息增益比，axis指定feature所在列
def cal_info_gain_ratio(dataset, axis=0):
    return cal_info_gain(dataset, axis) / cal_feat_entropy(dataset, axis)


if __name__ == '__main__':
    dataset = create_dataset()
    
    info_gains = []
    info_gain_ratios = []
    for i in range(dataset.shape[1]-1):
        info_gains.append(cal_info_gain(dataset, axis=i))
        info_gain_ratios.append(cal_info_gain_ratio(dataset, axis=i))
        print('特征\"{}\"的信息增益为{:.3f}，信息增益比为{:.3f}'
              .format(dataset.columns[i], info_gains[i], info_gain_ratios[i]))
    
    max_info_gain_i = info_gains.index(max(info_gains))
    max_info_gain_ratio_i = info_gain_ratios.index(max(info_gain_ratios))
    print('根据信息增益，最优特征为\"{}\"'.format(dataset.columns[max_info_gain_i]))
    print('根据信息增益比，最优特征为\"{}\"'.format(dataset.columns[max_info_gain_ratio_i]))
