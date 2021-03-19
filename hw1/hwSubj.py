import pandas as pd
import numpy as np
import math
data = pd.read_csv("test.csv", header=None)
data = data.iloc[:, 2:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
length = len(raw_data)
tdSet = np.empty((int(length/18), 9*18))

for i in range(length):
    j = int(i%18)
    tdSet[int(i/18)][j*9:(j+1)*9] = raw_data[i]

#归一化
tDataMean = np.mean(tdSet, axis=0)
std_x = np.std(tdSet, axis=0)
for i in range(len(tdSet)):
    for j in range(len(tdSet[i])):
        tdSet[i][j] = (tdSet[i][j]-tDataMean[j])/std_x[j]

ones = np.ones([len(tdSet), 1])
tdSet = np.concatenate((ones, tdSet), axis=1)

w = np.load("weight.npy")

y = np.dot(tdSet, w)

print(y)
