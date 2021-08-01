#! /usr/bin/python3

import csv
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
from numpy.lib.function_base import median
import pywt
from fastdtw import fastdtw
# import mlpy
from scipy.spatial.distance import euclidean
from scipy import stats
import sklearn.preprocessing as skprep
import numpy as np

res_factor = 5
smooth_factor = 15
trigger_sensitivity = 7
# gestures = ["leg_v", "leg_h", "leg_s", "hand_s"]

reference = {}
time = {}


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
    timeline = [float(i) for i in packets["Time"]]
    dBm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]

    # get 1/3 of data
    timeline = timeline[::res_factor]
    dBm = dBm[::res_factor]
    dBm = skprep.scale(
        dBm, axis=0, with_mean=True, with_std=True, copy=True
    )

    # Denoising
    w = pywt.Wavelet('haar')
    w_coef = pywt.wavedec(dBm, w)
    # sigma = mad(w_coef[-3])
    sigma = 0.8
    th = sigma * np.sqrt(2*np.log(len(dBm)))
    w_coef[1:] = (pywt.threshold(i, value=th, mode='soft')
                  for i in w_coef[1:])
    denoised_dBm = pywt.waverec(w_coef, 'haar')

    # smooth
    box = np.ones(smooth_factor)/smooth_factor
    denoised_dBm = np.convolve(denoised_dBm, box, mode='same')
    return denoised_dBm, timeline


def extract_ref(filename, gesture):
    ''' Extracts a single file of reference sequence for a specifc gesture
    '''
    denoised_dBm, timeline = extract_signal(filename)
    sublen = int(len(denoised_dBm) / 3)
    for i in range(0, 3):
        reference[gesture].append(denoised_dBm[sublen*i:sublen*(i+1)])
        time[gesture].append(timeline[sublen*i:sublen*(i+1)])


def extract_references():
    # list for reference signals sequences
    for g in gestures:
        reference[g] = []
        time[g] = []
    extract_ref("data/leg_vertical/leg_v_01.csv", "leg_v")
    extract_ref("data/leg_vertical/leg_v_02.csv", "leg_v")
    extract_ref("data/leg_horizontal/leg_h_01.csv", "leg_h")
    extract_ref("data/leg_horizontal/leg_h_02.csv", "leg_h")
    extract_ref("data/leg_step/leg_s_01.csv", "leg_s")
    extract_ref("data/leg_step/leg_s_02.csv", "leg_s")
    # extract_ref("data/leg_inf/leg_i_01.csv", "leg_i")
    # extract_ref("data/leg_inf/leg_i_02.csv", "leg_i")
    # extract_ref("data/leg_inf/leg_i_03.csv", "leg_i")
    extract_ref("data/hand_snatch/hand_s_01.csv", "hand_s")


def RSS_decode(signal):
    encode = "n"
    s_min = min(signal)
    s_max = max(signal)
    trigger = (s_max - s_min) / trigger_sensitivity
    prev = signal[0]
    stopwatch = 0
    sw_start = False
    i = 10
    while i < len(signal) and stopwatch < 250:
        # for i in range(10, len(signal), 10):
        if sw_start == True:
            stopwatch += 1
        curr = signal[i]
        diff = curr - prev
        if diff > trigger:
            sw_start = True
            c_state = "r"
        elif diff < -trigger:
            sw_start = True
            c_state = "f"
        else:
            c_state = "n"
        prev = curr
        i += 10
        if encode[-1] == c_state:
            continue
        else:
            encode += c_state
    return encode


def RSS_dtw(x, y):
    dist, _ = fastdtw(x, y, dist=euclidean)
    return dist


def classify_test(filename):
    denoised_dBm, _ = extract_signal(filename)
    test_cases = []
    sublen = int(len(denoised_dBm) / 3)
    for i in range(0, 3):
        test_cases.append(denoised_dBm[sublen*i:sublen*(i+1)])

    for case in test_cases:
        best = Inf
        tag = None
        num = None
        for g in gestures:
            # print(g)
            sum = 0
            for c, subref in enumerate(reference[g]):
                # print(subref)
                # print(case)
                curr = RSS_dtw(case, subref)
                print(g, c, curr)
                sum += curr
            avg = sum / len(reference[g])
            if avg < best:
                best = avg
                tag = g
        print("choose: ", tag, best)
        # break


def show_signal(gesture):
    plt.figure()
    print(gesture, len(reference[gesture]))
    for i in range(0, len(reference[gesture])):
        print(RSS_decode(reference[gesture][i]))
        my_min = min(len(time[gesture][i]), len(reference[gesture][i]))
        plt.subplot(len(reference[gesture]), 1, i+1)
        plt.plot(time[gesture][i][:my_min], reference[gesture][i][:my_min])
        plt.xlabel('Time (s)')
    plt.savefig(gesture+".png")


if __name__ == "__main__":
    extract_references()
    # classify_test("data/testset/vhs01.csv")
    # for g in gestures:
    #     show_signal(g)
