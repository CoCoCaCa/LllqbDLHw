# python demo
# 记录在作业中所使用的python相关代码的说明

import numpy as np

X = np.array([[1, 2], [3, 4], [1, 2]])
specified_column = np.arange(X.shape[1])
X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
print(X[[0, 1], :])
