#! /usr/bin/python3
import csv
import pywt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprep

def denoise_signal(signal):
    w = pywt.Wavelet('haar')
    w_coef = pywt.wavedec(signal, w)
    # sigma = mad(w_coef[-3])
    sigma = 0.8
    th = sigma * np.sqrt(2*np.log(len(signal)))
    w_coef[1:] = (pywt.threshold(i, value=th, mode='soft')
                for i in w_coef[1:])
    return pywt.waverec(w_coef, 'haar')

def smoothen_signal(signal, factor):
    box = np.ones(factor)/factor
    return np.convolve(signal, box, mode='same')

def RSS_decode(signal, trigger_factor):
    encode = "n"
    s_min = min(signal)
    s_max = max(signal)
    trigger = (s_max - s_min) / trigger_factor

    prev = signal[0]
    stopwatch = 0
    sw_start = False
    for i in range(10, len(signal), 10):
        if sw_start:
            stopwatch += 1
        
        curr = signal[i]
        diff = curr - prev
        prev = curr
        if diff > trigger:
            sw_start = True
            state = "r"
        elif diff < trigger:
            sw_start = True
            state = "f"
        else:
            state = "n"
        # Check state change
        if encode[-1] == state:
            continue
        else:
            encode += state
    return encode


class RSSalut:
    # factors
    res_factor = 5
    smooth_factor = 15

    # data
    gestures = ["leg_v", "leg_h", "leg_s", "hand_s"]
    reference = {}
    timeline = {}
    encoded = {}

    def __init__(self):
        # Initialise reference data structs
        for g in self.gestures:
            # print(g)
            self.reference[g] = []
            self.timeline[g] = []
            self.encoded[g] = []

        # Extract reference
        self.extract_references()

        
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
        timeline = [float(i) for i in packets["Time"]]
        dBm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]
        
        # lower resolution
        timeline = timeline[::self.res_factor]
        dBm = dBm[::self.res_factor]
        dBm = skprep.scale(
                dBm, axis=0, with_mean=True, with_std=True, copy=True
            )

        # Process signal
        dBm = denoise_signal(dBm)
        dBm = smoothen_signal(dBm, self.smooth_factor)
        return dBm, timeline

    def extract_ref(self, filename, gesture):
        dBm, timeline = self.extract_signal(filename)
        sublen = int(len(dBm) / 3)
        for i in range(0, 3):
            self.reference[gesture].append(dBm[sublen*i:sublen*(i+1)])
            self.timeline[gesture].append(timeline[sublen*i:sublen*(i+1)])

    def extract_references(self):
        self.extract_ref("data/leg_vertical/leg_v_01.csv", "leg_v")
        self.extract_ref("data/leg_vertical/leg_v_02.csv", "leg_v")
        self.extract_ref("data/leg_horizontal/leg_h_01.csv", "leg_h")
        self.extract_ref("data/leg_horizontal/leg_h_02.csv", "leg_h")
        self.extract_ref("data/leg_step/leg_s_01.csv", "leg_s")
        self.extract_ref("data/leg_step/leg_s_02.csv", "leg_s")
        # self.extract_ref("data/leg_inf/leg_i_01.csv", "leg_i")
        # self.extract_ref("data/leg_inf/leg_i_02.csv", "leg_i")
        # self.extract_ref("data/leg_inf/leg_i_03.csv", "leg_i")
        self.extract_ref("data/hand_snatch/hand_s_01.csv", "hand_s")

    def show_signal(self, gesture):
        plt.figure()
        for i in range(0, len(self.reference[gesture])):
            my_min = min(len(self.timeline[gesture][i]), len(self.reference[gesture][i]))
            plt.subplot(len(self.reference[gesture]), 1, i+1)
            plt.plot(self.timeline[gesture][i][:my_min], self.reference[gesture][i][:my_min])
            plt.xlabel('Time (s)')
        plt.savefig(gesture+".png")

    def show_all_signals(self):
        for g in self.gestures:
            self.show_signal(g)

if __name__ == "__main__":
    rssalut = RSSalut()
    rssalut.show_all_signals()