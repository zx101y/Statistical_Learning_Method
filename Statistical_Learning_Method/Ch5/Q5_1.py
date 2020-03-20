import math


# 树的节点类
class Node(object):
    def __init__(self, is_leaf=False, title=None):
        self.is_leaf = is_leaf
        self.title = title  # 非叶节点为分类特征名，叶节点为类别标签
        self.child_list = {}  # 格式为 特征取值：子节点引用


# 决策树类
class DecisionTree(object):  
    def __init__(self):
        self.tree = Node()  # 根节点
        
    # 计算数据集的熵
    def cal_entropy(self, dataset):
        n = len(dataset)
        label_count = {}
        entropy = 0
        
        for data in dataset:
            label = data[-1]  # 最后一项为标签
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        
        for label in label_count:
            p = label_count[label] / n
            entropy += (-p * math.log(p, 2))  # 使用2为底的对数
        
        return entropy
    
    # 计算数据集在某个特征下的条件熵
    def cal_cond_entropy(self, dataset, axis=0):
        n = len(dataset)
        divisive_lists = {}  # 把每个样本按指定的特征分到不同的list里面
        cond_entropy = 0
        
        for data in dataset:
            feature = data[axis]
            if feature in divisive_lists:
                divisive_lists[feature].append(data)
            else:
                divisive_lists[feature] = [data]
                
        for feature in divisive_lists:
            entropy = self.cal_entropy(divisive_lists[feature])
            cond_entropy += len(divisive_lists[feature]) / n * entropy
        
        return cond_entropy
    
    # 计算数据集关于某个特征的熵
    def cal_feat_entropy(self, dataset, axis=0):
        n = len(dataset)
        feat_count = {}
        feat_entropy = 0
        
        for data in dataset:
            feature = data[axis]
            if feature in feat_count:
                feat_count[feature] += 1
            else:
                feat_count[feature] = 1
        
        for feature in feat_count:
            p = feat_count[feature] / n
            feat_entropy += (-p * math.log(p, 2))  # 使用2为底的对数
        
        return feat_entropy
    
    # 计算信息增益
    def cal_info_gain(self, dataset, axis=0):
        return self.cal_entropy(dataset) - self.cal_cond_entropy(dataset, axis)
    
    # 计算信息增益比
    def cal_info_gain_ratio(self, dataset, axis=0):
        return self.cal_info_gain(dataset, axis) / self.cal_feat_entropy(dataset, axis)
    
    # 训练决策树
    def train(self, dataset, titles):
        self._train_ope(dataset, titles, self.tree)
    
    # 训练决策树，递归
    def _train_ope(self, dataset, titles, node):
        # 只剩一个样本，已到达叶节点
        if len(dataset) == 1:
            node.is_leaf = True
            node.title = dataset[0][-1]
            return
        
        # 找出信息增益比最高的特征
        info_gain_ratios = []
        for i in range(len(titles)-1):
            info_gain_ratios.append(self.cal_info_gain_ratio(dataset, axis=i))
        max_ratio = max(info_gain_ratios)  # 最大信息增益比
        title_i = info_gain_ratios.index(max_ratio)  # 对应的特征名下标
        title = titles[title_i]  # 对应的特征名
        
        # 信息增益为0，说明当前节点下所有样本均为同一类，已到达子节点
        if max_ratio == 0:
            node.is_leaf = True
            node.title = dataset[0][-1]
            return
        
        # 根据找到的特征对数据集进行分类
        div_dataset = {}
        for data in dataset:
            data = data[:title_i] + data[title_i+1:]  # 去掉已使用的特征值
            if data[title_i] in div_dataset:
                data[title_i].append(data)
            else:
                data[title_i] = [data]
        
        # 更新当前节点，并对它的每个子节点递归分类
        node.is_leaf = False
        node.title = title
        for div_data_key in div_dataset:
            div_data = div_dataset[div_data_key]
            new_node = Node()
            self._train_ope(div_data, titles[:title_i]+titles[title_i+1:], new_node)  # 去掉已使用的特征
            node.child_list[div_data_key] = new_node


if __name__ == '__main__':
    # 表5.1的数据
    dataset = [['青年', '否', '否', '一般', '否'],
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
               ['老年', '否', '否', '一般', '否'],]
    titles = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    
    # 创建并训练决策树
    deci_tree = DecisionTree()
    deci_tree.train(dataset, titles)
    