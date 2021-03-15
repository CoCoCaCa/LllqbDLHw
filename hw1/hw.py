import sys
import pandas as pd
import numpy as np
import math

data = pd.read_csv("train.csv", encoding="big5")
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()
#print(raw_data)

month_data = {}
for month in range(12):
    #18行，480列的空数组
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

#归一化
tDataMean = np.mean(tData, axis=0)
std_x = np.std(tData, axis=0)
for i in range(len(tData)):
    for j in range(len(tData[i])):
        tData[i][j] = (tData[i][j]-tDataMean[j])/std_x[j]

tdSet = tData[:math.floor(len(tData)*0.8), :]
lbSet = label[:math.floor(len(label)*0.8), :]

tdValidation = tData[math.floor(len(tData)*0.8):, :]
lbValidation = label[math.floor(len(label)*0.8):, :]

print(len(tdSet[0]))
