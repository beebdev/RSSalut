#! /usr/bin/python3

import csv
import matplotlib.pyplot as plt
from numpy.lib.function_base import median
import pywt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy import stats
import numpy as np


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


headers = []
packets = {}

# Extract data
with open("data/leg_up_down.csv") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    for c, row in enumerate(reader):
        if c == 0:
            for field in row:
                packets[field] = []
                headers.append(field)
        else:
            for nf, value in enumerate(row):
                packets[headers[nf]].append(value)
    timeline = [float(i) for i in packets["Time"]]
    dbm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]

# Denoise
w = pywt.Wavelet('haar')
w_coef = pywt.wavedec(dbm, w)
# sigma = mad(w_coef[-1])
sigma = 0.7
print(sigma)
thresh = sigma * np.sqrt(2*np.log(len(dbm)))
w_coef[1:] = (pywt.threshold(i, value=thresh, mode='soft') for i in w_coef[1:])
datarec = pywt.waverec(w_coef, 'haar')


'''DTW'''
sublen = int(len(datarec) / 3)
l0 = [-51, -52, -53, -52]*(int(sublen/4))
l1 = datarec[:sublen]
l2 = datarec[sublen:sublen*2]
l3 = datarec[sublen*2:sublen*3]
distance, path = fastdtw(l1, l2, dist=euclidean)
print(distance)
distance, path = fastdtw(l1, l3, dist=euclidean)
print(distance)
distance, path = fastdtw(l2, l3, dist=euclidean)
print(distance)
distance, path = fastdtw(l0, l1, dist=euclidean)
print(distance)
distance, path = fastdtw(l0, l2, dist=euclidean)
print(distance)
distance, path = fastdtw(l0, l3, dist=euclidean)
print(distance)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(timeline[:sublen], l1)
plt.xlabel('Time (s)')
plt.ylabel('RSSI (dBm)')
plt.title("l1")
plt.subplot(3, 1, 2)
plt.plot(timeline[sublen:2*sublen], l2)
plt.xlabel('Time (s)')
plt.ylabel('RSSI (dBm)')
plt.title("l2")
plt.subplot(3, 1, 3)
plt.plot(timeline[2*sublen:sublen*3], l3)
plt.xlabel('Time (s)')
plt.ylabel('RSSI (dBm)')
plt.title("l3")
plt.tight_layout()
plt.show()

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(timeline, dbm)
# plt.xlabel('Time (s)')
# plt.ylabel('RSSI (dBm)')
# plt.title("Raw signal")
# plt.subplot(2, 1, 2)
# plt.plot(timeline, datarec[:-1])
# plt.xlabel('Time (s)')
# plt.ylabel('RSSI (dBm)')
# plt.title("De-noised signal using wavelet techniques")
# plt.tight_layout()
# plt.show()
