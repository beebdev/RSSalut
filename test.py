#! /usr/bin/python3
import csv
import pywt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprep


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def denoise_signal(signal):
    w = pywt.Wavelet('haar')
    w_coef = pywt.wavedec(signal, w)
    # sigma = mad(w_coef[-2])
    sigma = 0.8
    th = sigma * np.sqrt(2*np.log(len(signal)))
    w_coef[1:] = (pywt.threshold(i, value=th, mode='soft')
                  for i in w_coef[1:])
    return pywt.waverec(w_coef, 'haar')


def smoothen_signal(signal, factor):
    box = np.ones(factor)/factor
    return np.convolve(signal, box, mode='same')


def gesture_decode(signal, trigger_factor, pps):
    decode = "n"
    s_min = min(signal)
    s_max = max(signal)
    trigger = (s_max - s_min) / trigger_factor
    n_timer = 0

    inc = 45
    prev = signal[0]
    for i in range(inc, len(signal), inc):
        curr = signal[i]
        diff = curr - prev
        prev = curr

        if diff > trigger:
            state = "r"
            n_timer = 0
        elif diff < -trigger:
            state = "f"
            n_timer = 0
        else:
            state = "n"
            n_timer += inc

        # Check state change
        if decode[-1] == state:
            continue
        else:
            # print(i)
            if state != "n" or n_timer > 1.3*pps:
                decode += state
                n_timer = 0
    if decode[-1] != "n":
        decode += "n"
    return decode


def next_big_thing(signal, start, trigger_factor):
    """
    Returns the index of when the next major change happens
    """
    s_min = min(signal)
    s_max = max(signal)
    trigger = (s_max - s_min) / trigger_factor

    prev = signal[start]
    for i in range(start+10, len(signal), 10):
        curr = signal[i]
        diff = curr - prev
        prev = curr
        if diff > trigger:
            return i
        elif diff < -trigger:
            return i
    return -1


class RSSalut:
    # factors
    res_factor = 3
    smooth_factor = 12
    trig_sensitivity = 5

    # macros
    pps = 0

    # data
    nGestures = 3
    nPerGesture = 5
    # gestures = ["gesture", "leg_h", "leg_s", "hand_s"]
    reference = []
    timeline = []
    encoded = []

    def __init__(self):
        # Initialise reference data structs
        for i in range(0, self.nGestures):
            self.reference.append([])
            self.timeline.append([])
            self.encoded.append([])

        # Extract reference
        self.extract_references()
        self.pps = int(
            len(self.reference[0][0])/self.timeline[0][0][-1])
        print(self.pps)

    def extract_signal(self, filename):
        # Open csv data file
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
        timeline = [float(i) for i in packets["Time"]][:]
        dBm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]

        # lower resolution
        sublen = int((len(dBm) / 30))
        timeline = timeline[sublen*2:-sublen*3:self.res_factor]
        dBm = dBm[sublen*2:-sublen*3:self.res_factor]
        dBm = skprep.scale(
            dBm, axis=0, with_mean=True, with_std=True, copy=True
        )

        # Process signal
        dBm = denoise_signal(dBm)
        dBm = smoothen_signal(dBm, self.smooth_factor)
        return dBm, timeline

    def extract_ref(self, filename, gesture):
        dBm, timeline = self.extract_signal(filename)
        sublen = int((len(dBm) / 30)*1.5)
        self.reference[gesture].append(dBm)  # [sublen:-sublen*2])
        self.timeline[gesture].append(timeline)  # [sublen:-sublen*2])
        # for i in range(0, 3):
        #     self.reference[gesture].append(dBm[sublen*i:sublen*(i+1)])
        #     self.timeline[gesture].append(timeline[sublen*i:sublen*(i+1)])

    def extract_references(self):
        dir_path = "data/gesture"
        for i in range(0, self.nGestures):
            gdir_path = dir_path + str(i) + "/"
            for j in range(1, self.nPerGesture+1):
                file_path = gdir_path + "0" + str(j) + ".csv"
                self.extract_ref(file_path, i)

    def show_signal(self, gesture):
        plt.figure()
        for i in range(0, len(self.reference[gesture])):
            my_min = min(len(self.timeline[gesture][i]), len(
                self.reference[gesture][i]))
            plt.subplot(len(self.reference[gesture]), 1, i+1)
            plt.plot(self.timeline[gesture][i][:my_min],
                     self.reference[gesture][i][:my_min])
            plt.xlabel('Time (s)')
        plt.savefig(str(gesture)+".png")

    def show_all_signals(self):
        for i in range(0, self.nGestures):
            self.show_signal(i)

    def show_decode(self):
        for i in range(0, self.nGestures):
            for j in range(0, self.nPerGesture):
                print("Gesture", i, "; reference", j)
                sig = self.reference[i][j]
                # for t in range(0, len(self.reference[i][j]), 110):
                t = 0
                while t < len(sig):
                    res = next_big_thing(sig, t, 5)
                    if res == -1:
                        break
                    elif res < self.pps:
                        t += 30
                        continue
                    else:
                        t = res

                    code = gesture_decode(
                        sig[t-30:t+int(self.pps*4)], 5, self.pps)
                    print("\t@", t, "-", self.timeline[i][j][t], code)
                    t += self.pps*2
                    # print(RSS_decode(self.reference[i][j], self.trig_sensitivity))


if __name__ == "__main__":
    rssalut = RSSalut()
    rssalut.show_all_signals()
    rssalut.show_decode()
