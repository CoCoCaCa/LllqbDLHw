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


# 归一化
def _normalize(X, train=True, specifide_colum=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #   X: data to be processd
    #   train: 'True' when processing training data, 'False' for testing data
    #   specifide_colum: indexes of the columns that will be normalized. If 'None', all columns will be normalized
