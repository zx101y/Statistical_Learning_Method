import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


def read_dataset():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    # 把特征和标签连接起来
    dataset = np.concatenate((data[:, :2], target.reshape(-1, 1)), axis=1)
    
    return np.concatenate((dataset[0:5], dataset[50:55], dataset[100:105]), axis=0)


class KdTree(object):
    class Node(object):
        def __init__(self):
            self.data = None
            self.parent = None
            self.left_child = None
            self.right_child = None
            self.feature = None
    
    def __init__(self, dataset):
        self.root = self.Node()
        self.n_layers = 1
        self.construct(dataset, node=self.root, layer=1)
    
    def construct(self, dataset, node, layer):
        feature = (layer - 1) % (dataset.shape[1] - 1)  # 最后一列是标签
        dataset = dataset[dataset[:, feature].argsort()]  # 按照第feature个特征对各样本进行排序
        mid = dataset.shape[0] // 2
        
        if layer > self.n_layers:
            self.n_layers = layer
        
        node.data = dataset[mid]
        node.feature = feature
        
        if mid > 0:
            node.left_child = self.Node()
            node.left_child.parent = node
            self.construct(dataset[:mid], node.left_child, layer+1)
        if mid < dataset.shape[0] - 1:
            node.right_child = self.Node()
            node.right_child.parent = node
            self.construct(dataset[mid+1:], node.right_child, layer+1)
    
    def print_out(self):
        queue = [self.root]
        
        for i in range(self.n_layers):
            queue_temp = []
            
            while len(queue) > 0:
                node = queue.pop(0)
                if node is not None:
                    queue_temp.extend([node.left_child, node.right_child])
                    print(node.data, end=' ')
                else:
                    print('None', end=' ')
            print()
            
            queue = queue_temp
    
    def predict(self, data, k):
        points = []
        dists = []
        
        self.search(self.root, points, dists, data, k)
        
        label_count = {}
        for p in points:
            if p[-1] in label_count:
                label_count[p[-1]] += 1
            else:
                label_count[p[-1]] = 1
        label_count_reverse = {v:k for k, v in label_count.items()}
        
        return label_count_reverse.get(max(label_count.values()))
    
    def search(self, node, points, dists, data, k):
        if node == None:
            return
        
        if data[node.feature] < node.data[node.feature]:
            near_node = node.left_child
            far_node = node.right_child
        else:
            near_node = node.right_child
            far_node = node.left_child
        
        self.search(near_node, points, dists, data, k)
        
        if len(points) == k and max(dists) <= abs(data[node.feature] - node.data[node.feature]):
            return
        
        dist1 = self.dist(node.data[:-1], data)
        if len(points) < k:
            points.append(node.data)
            dists.append(dist1)
        elif dist1 < max(dists):
            i = dists.index(max(dists))
            points[i] = node.data
            dists[i] = dist1
        
        self.search(far_node, points, dists, data, k)
        
    
    def dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)


if __name__ == '__main__':
    dataset = read_dataset()
    
    kd_tree = KdTree(dataset)
    kd_tree.print_out()
    
    test_data = np.array([[4.6, 3.5, -1], [6.1, 2.9, -1], [6.8, 3.2, -1]])
    for i in range(len(test_data)):
        test_data[i, -1] = kd_tree.predict(test_data[i, :-1], 3)
    
    plt.scatter(dataset[dataset[:, -1]==0, 0], dataset[dataset[:, -1]==0, 1], color='r')
    plt.scatter(dataset[dataset[:, -1]==1, 0], dataset[dataset[:, -1]==1, 1], color='g')
    plt.scatter(dataset[dataset[:, -1]==2, 0], dataset[dataset[:, -1]==2, 1], color='b')
    plt.scatter(test_data[test_data[:, -1]==0, 0], test_data[test_data[:, -1]==0, 1], color='r', marker='*')
    plt.scatter(test_data[test_data[:, -1]==1, 0], test_data[test_data[:, -1]==1, 1], color='g', marker='*')
    plt.scatter(test_data[test_data[:, -1]==2, 0], test_data[test_data[:, -1]==2, 1], color='b', marker='*')
    plt.show()
    