import numpy as np
np.random.seed(0)

X_train_fpath = 'data/X_train'
Y_train_fpath = "data/Y_train"
X_test_fpath = "data/X_test"

with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

with open(X_test_fpath) as f:
    next(f)
    data = []
    for line in f:
        data.append(line.strip('\n').split(',')[1:])
    X_test = np.array(data)

