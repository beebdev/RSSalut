#! /usr/bin/python3

import csv
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
from numpy.lib.function_base import median
import pywt
from fastdtw import fastdtw
import mlpy
from scipy.spatial.distance import euclidean
from scipy import stats
import sklearn.preprocessing as skprep
import numpy as np

reference = {}


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def extract_signal(filename):
    with open(filename) as csv_file:
        headers = []
        packets = {}
        reader = csv.reader(csv_file, delimiter=",")
        for c, row in enumerate(reader):
            if c == 0:
                for field in row:
                    packets[field] = []
                    headers.append(field)
            else:
                for nf, value in enumerate(row):
                    packets[headers[nf]].append(value)
    # Timeline and reference
    # timeline = [float(i) for i in packets["Time"]]
    dBm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]

    # Denoising
    w = pywt.Wavelet('haar')
    w_coef = pywt.wavedec(dBm, w)
    # sigma = mad(w_coef[-3])
    sigma = 0.8
    th = sigma * np.sqrt(2*np.log(len(dBm)))
    w_coef[1:] = (pywt.threshold(i, value=th, mode='soft')
                  for i in w_coef[1:])
    denoised_dBm = pywt.waverec(w_coef, 'haar')
    return denoised_dBm


def extract_ref(filename, gesture):
    ''' Extracts a single file of reference sequence for a specifc gesture
    '''
    denoised_dBm = extract_signal(filename)
    sublen = int(len(denoised_dBm) / 3)
    for i in range(0, 3):
        reference[gesture].append(denoised_dBm[sublen*i:sublen*(i+1)])


def extract_references():
    # list for reference signals sequences
    reference["leg_v"] = []
    # reference["leg_h"] = []
    reference["leg_s"] = []
    reference["leg_i"] = []
    reference["hand_s"] = []
    extract_ref("data/leg_vertical/leg_v_01.csv", "leg_v")
    extract_ref("data/leg_vertical/leg_v_02.csv", "leg_v")
    # extract_ref("data/leg_horizontal/leg_h_01.csv", "leg_h")
    # extract_ref("data/leg_horizontal/leg_h_02.csv", "leg_h")
    extract_ref("data/leg_step/leg_s_01.csv", "leg_s")
    extract_ref("data/leg_step/leg_s_02.csv", "leg_s")
    extract_ref("data/leg_inf/leg_i_01.csv", "leg_i")
    extract_ref("data/leg_inf/leg_i_02.csv", "leg_i")
    extract_ref("data/leg_inf/leg_i_03.csv", "leg_i")
    extract_ref("data/hand_snatch/hand_s_01.csv", "hand_s")


def RSS_dtw(x, y):
    dist, _ = fastdtw(x, y, dist=euclidean)
    return dist


def classify_test(filename):
    denoised_dBm = extract_signal(filename)
    test_cases = []
    sublen = int(len(denoised_dBm) / 3)
    for i in range(0, 3):
        test_cases.append(denoised_dBm[sublen*i:sublen*(i+1)])

    for case in test_cases:
        best = Inf
        tag = None
        for k, v in reference.items():
            for l in v:
                curr = RSS_dtw(case, l)
                if curr < best:
                    tag = k
                    best = curr
        print(tag, best)


if __name__ == "__main__":
    extract_references()
    # print(reference)
    classify_test("data/testset/vhs01.csv")

    # print("Same")
    # RSS_dtw(reference["leg_v"][0], reference["leg_v"][1])
    # RSS_dtw(reference["leg_h"][0], reference["leg_h"][1])
    # RSS_dtw(reference["leg_s"][0], reference["leg_s"][1])
    # RSS_dtw(reference["leg_i"][0], reference["leg_i"][1])
    # RSS_dtw(reference["leg_i"][0], reference["leg_i"][2])
    # RSS_dtw(reference["leg_i"][0], reference["leg_i"][4])
    # RSS_dtw(reference["hand_s"][0], reference["hand_s"][1])
    # print("Different")
    # RSS_dtw(reference["leg_v"][0], reference["leg_h"][0])
    # RSS_dtw(reference["leg_v"][0], reference["leg_s"][1])
    # RSS_dtw(reference["leg_v"][0], reference["leg_i"][0])
    # RSS_dtw(reference["leg_h"][0], reference["leg_s"][0])
    # RSS_dtw(reference["leg_h"][0], reference["leg_i"][0])
    # RSS_dtw(reference["leg_s"][0], reference["leg_i"][0])
    # RSS_dtw(reference["hand_s"][0], reference["leg_v"][0])

    # headers = []
    # packets = {}

    # # Extract data
    # with open("data/leg_step/leg_s_02.csv") as csv_file:
    #     reader = csv.reader(csv_file, delimiter=",")
    #     for c, row in enumerate(reader):
    #         if c == 0:
    #             for field in row:
    #                 packets[field] = []
    #                 headers.append(field)
    #         else:
    #             for nf, value in enumerate(row):
    #                 packets[headers[nf]].append(value)
    #     timeline = [float(i) for i in packets["Time"]]
    #     dbm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]

    # # Denoise
    # w = pywt.Wavelet('haar')
    # w_coef = pywt.wavedec(dbm, w)
    # # sigma = mad(w_coef[-3])
    # sigma = 0.8
    # print(sigma)
    # thresh = sigma * np.sqrt(2*np.log(len(dbm)))
    # w_coef[1:] = (pywt.threshold(i, value=thresh, mode='soft') for i in w_coef[1:])
    # datarec = pywt.waverec(w_coef, 'haar')

    # '''DTW'''
    # sublen = int(len(datarec) / 3)
    # l0 = [-51, -52, -53, -52]*(int(sublen/4))
    # l1 = datarec[:sublen]
    # l2 = datarec[sublen:sublen*2]
    # l3 = datarec[sublen*2:sublen*3]
    # distance, path = fastdtw(l1, l2, dist=euclidean)
    # print(distance)
    # distance, path = fastdtw(l1, l3, dist=euclidean)
    # print(distance)
    # distance, path = fastdtw(l2, l3, dist=euclidean)
    # print(distance)
    # distance, path = fastdtw(l0, l1, dist=euclidean)
    # print(distance)
    # distance, path = fastdtw(l0, l2, dist=euclidean)
    # print(distance)
    # distance, path = fastdtw(l0, l3, dist=euclidean)
    # print(distance)
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(timeline[:sublen], l1)
    # plt.xlabel('Time (s)')
    # plt.ylabel('RSSI (dBm)')
    # plt.title("l1")
    # plt.subplot(3, 1, 2)
    # plt.plot(timeline[sublen:2*sublen], l2)
    # plt.xlabel('Time (s)')
    # plt.ylabel('RSSI (dBm)')
    # plt.title("l2")
    # plt.subplot(3, 1, 3)
    # plt.plot(timeline[2*sublen:sublen*3], l3[:-1])
    # plt.xlabel('Time (s)')
    # plt.ylabel('RSSI (dBm)')
    # plt.title("l3")
    # plt.tight_layout()
    # plt.show()

    # # plt.figure()
    # # plt.subplot(2, 1, 1)
    # # plt.plot(timeline, dbm)
    # # plt.xlabel('Time (s)')
    # # plt.ylabel('RSSI (dBm)')
    # # plt.title("Raw signal")
    # # plt.subplot(2, 1, 2)
    # # plt.plot(timeline, datarec[:-1])
    # # plt.xlabel('Time (s)')
    # # plt.ylabel('RSSI (dBm)')
    # # plt.title("De-noised signal using wavelet techniques")
    # # plt.tight_layout()
    # # plt.show()
