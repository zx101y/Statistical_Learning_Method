from Info_gain import cal_entropy, cal_info_gain
from Create_dataset import create_dataset
import networkx as nx
import matplotlib.pyplot as plt


class DecisionTree(object):
    class Node(object):
        def __init__(self, num, is_leaf=None, label=None, div_feat=None):
            self.num = num  # 节点编号（根节点为0开始）
            self.is_leaf = is_leaf  # 是否是叶节点
            self.label = label  # 类属性（当前节点的样本集中，数量最多的label）
            self.div_feat = div_feat  # 下一步分类依据的特征（非叶节点独有）
            self.children = {}  # 子节点列表，格式为“特征取值：子节点”
            self.weighted_entropy = None  # 节点的加权熵（样本个数 * 经验熵）
    
    def __init__(self):
        self.root = self.Node(num=0)
        self.count = 1  # 计算当前的节点数
    
    # 训练决策树
    def train(self, dataset, epsilon):
        self.train_ope(dataset, epsilon, self.root)
    
    # 训练决策树的迭代执行函数
    def train_ope(self, dataset, epsilon, node):
        # 记录节点的加权熵
        node.weighted_entropy = cal_entropy(dataset) * len(dataset)
        
        # 对最后一列（label）进行计数，得到一个Series，然后取数量最大的索引（label）
        node.label = dataset.iloc[:, -1].value_counts().idxmax()
        
        # dataset中所有实例属于同一类
        if len(set(dataset.iloc[:, -1])) == 1:
            node.is_leaf = True
            
            return
        
        # 用于划分的特征集为空集（只剩label）
        if dataset.shape[1] == 1:
            node.is_leaf = True
            
            return
        
        # 找出信息增益最大的feature
        # cal_info_gain改成cal_info_gain_ratio，就变成C4.5算法
        info_gains = []
        for i in range(dataset.shape[1]-1):
            info_gains.append(cal_info_gain(dataset, axis=i))
        max_info_gain = max(info_gains)
        best_feature_idx = info_gains.index(max_info_gain)
        best_feature = dataset.columns[best_feature_idx]
        
        # 最大信息增益小于阈值
        if max_info_gain < epsilon:
            node.is_leaf = True
            
            return
        
        # 可以进一步分类
        node.is_leaf = False
        node.div_feat = best_feature
        
        # 统计该feature的所有取值，并建立子树
        feature_vals = set()
        for i in range(len(dataset)):
            feature_vals.add(dataset.iloc[i, best_feature_idx])
            
        for val in feature_vals:
            node.children[val] = self.Node(self.count)
            self.count += 1
            next_dataset = dataset.loc[dataset[best_feature]==val]  # 选取该feature为特定取值的所有行
            next_dataset = next_dataset.drop(columns=[best_feature])  # 删除该feature所在列
            self.train_ope(next_dataset, epsilon, node.children[val])  # 训练子树
    
    # 决策树剪枝
    def pruning(self, alpha):
        self.pruning_ope(alpha, self.root)
    
    # 决策树剪枝的迭代执行函数
    def pruning_ope(self, alpha, node):
        # 叶节点，直接返回加权熵和1（叶节点个数）
        if node.is_leaf:
            return node.weighted_entropy, 1
        
        # 非叶节点
        # 计算该节点下的叶节点数、叶节点的加权熵的和，获得剪枝前的损失函数值
        # 剪枝后节点会成为叶节点，按单个节点计算剪枝后的损失函数值
        weighted_entropy_sum = 0
        n_leaves = 0
        
        for child in node.children.values():
            temp1, temp2 = self.pruning_ope(alpha, child)
            weighted_entropy_sum += temp1
            n_leaves += temp2
        
        cost_before_pruning = weighted_entropy_sum + alpha * n_leaves
        cost_after_pruning = node.weighted_entropy + alpha
        
        # 剪枝会使损失减小，进行剪枝
        if cost_after_pruning < cost_before_pruning:
            node.is_leaf = True  # 变为叶节点
            node.div_feat = None  # 不再下分
            node.children = {}  # 子节点列表置空
            
            return node.weighted_entropy, 1
        # 剪枝不能优化，不剪枝，把叶节点的加权熵的和与叶节点数上传
        else:
            return weighted_entropy_sum, n_leaves
    
    
    # 用决策树进行预测
    def predict(self, data):
        node = self.root
        while not node.is_leaf:
            node = node.children[data[node.div_feat]]
        
        return node.label
    
    # 决策树可视化
    def visualization(self):
        dg = nx.DiGraph()  # 定义有向图
        node_queue = [self.root]  # 用一个队列来遍历决策树
        pos = {}
        row = 0
        
        while len(node_queue) > 0:
            col = 0
            node_queue_temp = []
            
            for node in node_queue:
                if node.is_leaf:
                    dg.add_node(node.num, val=node.label)  # 创建一个带label的叶节点
                else:
                    dg.add_node(node.num, val=node.div_feat)  # 创建一个带分类特征的非叶节点
                    for child_idx, child in node.children.items():
                        node_queue_temp.append(child)
                        # 创建该节点和所有子节点之间的边，边属性为分类特征的取值
                        dg.add_edge(node.num, child.num, val=child_idx)
                
                pos[node.num] = [row, col]
                col += 1
            
            node_queue = node_queue_temp
            row += 1
        
        # 绘制决策树
        nx.draw(dg, pos, node_size=1000)
        node_labels = nx.get_node_attributes(dg, 'val')  # 提取名为val的节点属性
        nx.draw_networkx_labels(dg, pos, labels=node_labels)
        nx.draw_networkx_edge_labels(dg, pos)
        plt.show()


if __name__ == '__main__':
    dataset = create_dataset(dataset_num=2)
    
    tree = DecisionTree()
    tree.train(dataset, 1e-5)
    
    tree.visualization()
    for i in range(len(dataset)):
        print(tree.predict(dataset.iloc[i]), end=' ')
    print()
    
    tree.pruning(alpha=3)
    
    tree.visualization()
    for i in range(len(dataset)):
        print(tree.predict(dataset.iloc[i]), end=' ')
    print()
