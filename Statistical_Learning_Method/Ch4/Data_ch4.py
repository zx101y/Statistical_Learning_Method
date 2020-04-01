import numpy as np
from sklearn import datasets


def read_dataset():
    x1 = np.array([1] * 5 + [2] * 5 + [3] * 5).reshape(-1, 1)
    x2 = np.array(['S', 'M', 'M', 'S', 'S',
                   'S', 'M', 'M', 'L', 'L',
                   'L', 'M', 'M', 'L', 'L']).reshape(-1, 1)
    y = np.array([-1, -1, 1, 1, -1,
                  -1, -1, 1, 1, 1,
                  1, 1, 1, 1, -1]).reshape(-1, 1)
    
    return np.concatenate([x1, x2, y], axis=1)
