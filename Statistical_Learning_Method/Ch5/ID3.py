from Info_gain import cal_info_gain
from Create_dataset import create_dataset
import networkx as nx
import matplotlib.pyplot as plt


class Node(object):
    def __init__(self, num, is_leaf=None, label=None, div_feat=None):
        self.num = num
        self.is_leaf = is_leaf
        self.label = label  # 类属性（叶节点独有）
        self.div_feat = div_feat  # 下一步分类依据的特征（非叶节点独有）
        self.children = {}  # 子节点列表，格式为“特征取值：子节点”


class DecisionTree(object):
    def __init__(self):
        self.tree = Node(num=0)
        self.count = 1  # 计算当前的节点数
    
    # 训练决策树
    def train(self, dataset, epsilon):
        self.train_ope(dataset, epsilon, self.tree)
    
    # 训练决策树的迭代执行函数
    def train_ope(self, dataset, epsilon, tree):
        # dataset中所有实例属于同一类
        if len(set(dataset.iloc[:, -1])) == 1:
            tree.is_leaf = True
            tree.label = dataset.iloc[0, -1]
            
            return
        
        # 用于划分的特征集为空集
        if dataset.shape[1] == 1:
            tree.is_leaf = True
            # 对最后一列（label）进行计数，得到一个Series，然后取数量最大的索引（label）
            tree.label = dataset.iloc[:, -1].value_counts().idxmax()
            
            return
        
        # 找出信息增益最大的feature
        info_gains = []
        for i in range(dataset.shape[1]-1):
            info_gains.append(cal_info_gain(dataset, axis=i))
        max_info_gain = max(info_gains)
        best_feature_idx = info_gains.index(max_info_gain)
        best_feature = dataset.columns[best_feature_idx]

        # # 找出信息增益比最大的feature（C4.5算法跟ID3算法的唯一区别）
        # info_gain_ratios = []
        # for i in range(dataset.shape[1] - 1):
        #     info_gain_ratios.append(cal_info_gain_ratio(dataset, axis=i))
        # max_info_gain = max(info_gain_ratios)
        # best_feature_idx = info_gain_ratios.index(max_info_gain)
        # best_feature = dataset.columns[best_feature_idx]
        
        # 最大信息增益小于阈值
        if max_info_gain < epsilon:
            tree.is_leaf = True
            # 对最后一列（label）进行计数，得到一个Series，然后取数量最大的索引（label）
            tree.label = dataset.iloc[:, -1].value_counts().idxmax()
            
            return
        
        # 可以进一步分类
        tree.is_leaf = False
        tree.div_feat = best_feature
        
        # 统计该feature的所有取值，并建立子树
        feature_vals = set()
        for i in range(len(dataset)):
            feature_vals.add(dataset.iloc[i, best_feature_idx])
            
        for val in feature_vals:
            tree.children[val] = Node(self.count)
            self.count += 1
            next_dataset = dataset.loc[dataset[best_feature]==val]  # 选取该feature为特定取值的所有行
            next_dataset = next_dataset.drop(columns=[best_feature])  # 删除该feature所在列
            self.train_ope(next_dataset, epsilon, tree.children[val])  # 训练子树
    
    # 决策树可视化
    def visualization(self):
        dg = nx.DiGraph()  # 定义有向图
        node_queue = [self.tree]  # 用一个队列来遍历决策树
        
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            
            if node.is_leaf:
                dg.add_node(node.num, val=node.label)  # 创建一个带label的叶节点
            else:
                dg.add_node(node.num, val=node.div_feat)  # 创建一个带分类特征的非叶节点
                for child_idx, child in node.children.items():
                    node_queue.append(child)
                    # 创建该节点和所有子节点之间的边，边属性为分类特征的取值
                    dg.add_edge(node.num, child.num, val=child_idx)
        
        # 绘制决策树
        pos = nx.spectral_layout(dg)
        nx.draw(dg, pos, node_size=2000)
        node_labels = nx.get_node_attributes(dg, 'val')  # 提取名为val的节点属性
        nx.draw_networkx_labels(dg, pos, labels=node_labels)
        nx.draw_networkx_edge_labels(dg, pos)
        plt.show()
    
    # 用决策树进行预测
    def predict(self, data):
        node = self.tree
        while not node.is_leaf:
            node = node.children[data[node.div_feat]]
        
        return node.label


if __name__ == '__main__':
    dataset = create_dataset()
    
    tree = DecisionTree()
    tree.train(dataset, 1e-5)
    tree.visualization()
    
    for i in range(len(dataset)):
        print(tree.predict(dataset.iloc[i]), end=' ')
    print()
