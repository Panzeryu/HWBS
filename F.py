# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn
import os
import linecache
import pandas as pd
import math
import datetime
from scipy.stats import stats
from scipy.fftpack import fft, ifft
from tqdm import tqdm


# 时间差计算函数
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


# os.makedirs('D:/TMD/picture/1')
# os.makedirs('D:/TMD/picture/2')
# os.makedirs('D:/TMD/picture/3')
# os.makedirs('D:/TMD/picture/4')
# os.makedirs('D:/TMD/picture/5')
# os.makedirs('D:/TMD/picture/6')
# os.makedirs('D:/TMD/picture/7')
# os.makedirs('D:/TMD/picture/8')


# -----super perimeter-----
fs = 4096
N = 256
step = 2
num_every_pic_contain = int(((500 - N) / step) + 1)
nrows = 100000
# -----super perimeter-----


startdate = datetime.datetime.now()                  # 获取当前时间
startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式
filepath_dataax = 'D:/TMD/data/huawei_data/Acc_x.txt'
filepath_dataay = 'D:/TMD/data/huawei_data/Acc_y.txt'
filepath_dataaz = 'D:/TMD/data/huawei_data/Acc_z.txt'
filepath_datagx = 'D:/TMD/data/huawei_data/Gyr_x.txt'
filepath_datagy = 'D:/TMD/data/huawei_data/Gyr_y.txt'
filepath_datagz = 'D:/TMD/data/huawei_data/Gyr_z.txt'
filepath_datala = 'D:/TMD/data/huawei_data/Label.txt'

dataax = pd.read_csv(filepath_or_buffer=filepath_dataax, header=None, sep=' ', dtype=np.float64, nrows=nrows)  # , nrows=nrows
dataay = pd.read_csv(filepath_or_buffer=filepath_dataay, header=None, sep=' ', dtype=np.float64, nrows=nrows)  # , nrows=nrows
dataaz = pd.read_csv(filepath_or_buffer=filepath_dataaz, header=None, sep=' ', dtype=np.float64, nrows=nrows)  # , nrows=nrows
datagy = pd.read_csv(filepath_or_buffer=filepath_datagy, header=None, sep=' ', dtype=np.float64, nrows=nrows)  # , nrows=nrows
datala = pd.read_csv(filepath_or_buffer=filepath_datala, header=None, sep=' ', dtype=np.float64, nrows=nrows)  # , nrows=nrows

total = len(dataax)
round = 10
# numeveryround=total/round

dataax = dataax.transpose()
dataay = dataay.transpose()
dataaz = dataaz.transpose()
datagy = datagy.transpose()
datala = datala.transpose()

picnum = 0
nannum = 0
fig, axx = plt.subplots()
Acc = []
Gyr = []
m1 = 0
m2 = 0
m3 = 0
m4 = 0
m5 = 0
m6 = 0
m7 = 0
m8 = 0

for i in tqdm(range(0, nrows, 1)):
    label = int(stats.mode(datala[i])[0][0])
    if label == 1:
        m1 += 1
        if m1 > 10:
            continue
    elif label == 2:
        m2 += 1
        if m2 > 10:
            continue
    elif label == 3:
        m3 += 1
        if m3 > 10:
            continue
    elif label == 4:
        m4 += 1
        if m4 > 10:
            continue
    elif label == 5:
        m5 += 1
        if m5 > 10:
            continue
    elif label == 6:
        m6 += 1
        if m6 > 10:
            continue
    elif label == 7:
        m7 += 1
        if m7 > 10:
            continue
    elif label == 8:
        m8 += 1
        if m8 > 10:
            continue

    for j in range(0, num_every_pic_contain, 1):
        ax = dataax[i][0 + j * step:N + j * step]
        ay = dataay[i][0 + j * step:N + j * step]
        az = dataaz[i][0 + j * step:N + j * step]
        an = ax * ax + ay * ay + az * az
        f = abs(fft(an, n=fs)) / (fs / 2)  # fft+归一化处理
        f1 = f[range(int(fs / 2))]  # 由于对称性，只取一半区间
        # f1 = np.array([0.001 if x < 0.05 else x for x in f1])
        Acc.append(f1)

    Acc = np.log10(Acc)
    Acc = (Acc - np.min(Acc)) / (np.max(Acc) - np.min(Acc))

    for j in range(0, num_every_pic_contain, 1):
        gy = datagy[i][0 + j * step:N + j * step]
        f = abs(fft(gy, n=fs)) / (fs / 2)  # fft+归一化处理
        f1 = f[range(int(fs / 2))]  # 由于对称性，只取一半区间
        # f1 = np.array([0.001 if x < 0.01 else x for x in f1])
        Gyr.append(f1)

    Gyr = np.log10(Gyr)
    Gyr = (Gyr - np.min(Gyr)) / (np.max(Gyr) - np.min(Gyr))

    pic = np.concatenate([Acc, Gyr])
    axx.imshow(pic, aspect='equal')
    height, width = pic.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    plt.imshow(pic)
    plt.savefig('D:/TMD/picture/%d/%d-%d.png' % (label, label, i+1), dpi=300)
    # plt.show()
    plt.clf()
    plt.close('all')
    del Acc
    del Gyr
    Acc = []
    Gyr = []
    picnum += 1
    if picnum % 10 == 0:
        print('number of picture is: %d' % picnum)


enddate = datetime.datetime.now()  # 获取当前时间
enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式

print('start date ', startdate)
print('end date ', enddate)
print('Time ', subtime(startdate, enddate))  # enddate > startdate
print('done')

