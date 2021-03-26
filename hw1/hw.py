import sys
import pandas as pd
import numpy as np
import math

data = pd.read_csv("train.csv", encoding="big5")
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
# print(raw_data)

month_data = {}
for month in range(12):
    # 18行，480列的空数组
    sample = np.empty([18, 480])
    for day in range(20):
        spMin = day*24
        spMax = spMin+24
        rdMin = (20*month+day)*18
        rdMax = rdMin+18
        sample[:, spMin:spMax] = raw_data[rdMin:rdMax, :]
    month_data[month] = sample

tData = np.empty([12*471, 18*9])
label = np.empty([12*471, 1])
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            tData[hour+day*24+month*471] = month_data[month][:, hour+day*24:hour+day*24+9].reshape(1, -1)
            label[hour+day*24+month*471] = month_data[month][9, hour+day*24+9]

# 归一化
tDataMean = np.mean(tData, axis=0)
std_x = np.std(tData, axis=0)
for i in range(len(tData)):
    for j in range(len(tData[i])):
        tData[i][j] = (tData[i][j]-tDataMean[j])/std_x[j]

tdSet = tData[:math.floor(len(tData)*0.8), :]
lbSet = label[:math.floor(len(label)*0.8), :]

tdValidation = tData[math.floor(len(tData)*0.6):, :]
lbValidation = label[math.floor(len(label)*0.6):, :]

dim = 18*9 + 1
# w = np.zeros([dim, 1])
w = np.load("weight.npy")
y_lbSet = np.zeros([len(lbSet), 1])
tdSet = np.concatenate((np.ones([len(tdSet), 1]), tdSet), axis=1).astype(float)
tdValidation = np.concatenate((np.ones([len(tdValidation), 1]), tdValidation), axis=1).astype(float)
adagrad = np.zeros([dim, 1])

def training(tdSet, lbSet, w, y_lbSet, adagrad):
    iter_time = 200
    eps = 0.0000000001
    learning_rate = 0.02
    for t in range(iter_time):
        y_lbSet = np.dot(tdSet, w) - lbSet
        loss = np.power(y_lbSet, 2)
        loss = np.sum(loss) / len(tdSet)
        loss = np.sqrt(loss)
        if (t % 100 == 0):
            print("迭代的次数：%i ， 损失值：%f" % (t, loss))
        gradient = np.dot(tdSet.transpose(), y_lbSet) / (loss * len(tdSet))
        adagrad = adagrad + (gradient ** 2)
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', w)

def testing(tdSet, lbSet, y_lbSet):
    y = np.dot(tdSet, w)
    y_lbSet = y - lbSet
    loss = np.power(y_lbSet, 2)
    loss = np.sum(loss) / len(tdSet)
    loss = np.sqrt(loss)
    print("测试数据 ， 损失值：%f" % (loss))
    print(np.concatenate((y, lbSet), axis=1))
    # print(y)
    # print(lbSet)

testing(tdValidation, lbValidation, y_lbSet)

