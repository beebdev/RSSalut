#! /usr/bin/python3

import csv
import pywt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skprep
from scipy.signal import savgol_filter

gestureNames = [u"\u001b[32mup_down\u001b[0m",
                "\u001b[33mdown_up\u001b[0m",
                "\u001b[34mdown_hold_up\u001b[0m",
                "No match"]
gestureCodes = [
    ["srfne", "srfe", "srnfne"],
    ["sfrnfe", "sfrne", "srfrfne", "sfrfe", "srfrne"],
    ["sfnrne", "sfnre"]]
datapath = "data/"


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variability of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def which_gesture(code):
    for gID, gesture in enumerate(gestureCodes):
        if code in gesture:
            return gID
    return -1


class RSSalut:
    # class attributes
    timeline = None
    dBm = None
    pps = 0
    marks = []
    timer = [0, 0, 0]
    filename = ""
    result = []

    # class functions
    def __init__(self, filename):
        self.filename = filename
        self.load_signal(filename)
        self._denoise()
        self._smoothen()
        # print(self.pps)
        self.event_parser()
        self.plot_dBm()

    def load_signal(self, filename):
        """ Loads raw WiFi data
        """
        # Extract data from file
        with open(filename) as csv_f:
            headers = []
            packets = {}
            reader = csv.reader(csv_f, delimiter=",")
            for c, row in enumerate(reader):
                if c == 0:  # header row
                    for field in row:
                        packets[field] = []
                        headers.append(field)
                else:  # data row
                    for nf, value in enumerate(row):
                        packets[headers[nf]].append(value)
            csv_f.close()

        # Timeline and reference
        timeline = [float(i) for i in packets["Time"]][:]
        dBm = [int(i[:-3]) for i in packets["Signal strength (dBm)"]]

        # Lower resolution (down sampling)
        sublen = int((len(dBm) / 30))
        # self.timeline = timeline[sublen*2:-sublen*3:3]
        # self.dBm = dBm[sublen*2:-sublen*3:3]
        self.timeline = timeline[sublen:-sublen:3]
        self.dBm = dBm[sublen:-sublen:3]
        self.pps = int(len(self.timeline) /
                       (self.timeline[-1]-self.timeline[0]))

    def _denoise(self):
        """ Denoises the dBm signal with DWT
        """
        w = pywt.Wavelet('haar')
        w_coef = pywt.wavedec(self.dBm, w)
        # sigma = mad(w_coef[-2])
        sigma = 1.1
        thresh = sigma * np.sqrt(2*np.log(len(self.dBm)))
        w_coef[1:] = (pywt.threshold(i, value=thresh, mode='soft')
                      for i in w_coef[1:])
        self.dBm = pywt.waverec(w_coef, 'haar')

        # Trim to match size of timeline and dBm
        if len(self.dBm) != len(self.timeline):
            shorty = min(len(self.dBm), len(self.timeline))
            self.dBm = self.dBm[:shorty]

    def _smoothen(self):
        """ Smoothens the dBm signal
        """
        self.dBm = savgol_filter(self.dBm, 101, 5)

    def event_parser(self):
        """ Parses the signal from the start of the signal.
            This is done by sliding a windows across the whole signal.
            In the window, when an RSS change triggers an event, the signal
            will be decoded into N/F/R based on rising/falling edge or no change
        """
        dBm_range = max(self.dBm) - min(self.dBm)
        window_size = self.pps*3
        window_slide = self.pps
        curr_time = self.timeline[0]
        print(self.timeline[0], self.timeline[window_size])
        for i in range(0, len(self.dBm), window_slide):
            w_dBm = self.dBm[i:i+window_size]
            w_time = self.timeline[i:i+window_size]

            # Determine if this section is flat
            w_range = max(w_dBm) - min(w_dBm)
            if w_range / dBm_range < 0.2:
                continue

            # Event trigger for this window
            w_trigger = w_range/5
            w_interval = int(self.pps/6)
            base = (0, w_dBm[0])
            for j in range(w_interval, min(window_size, len(w_dBm[w_interval:])), w_interval):
                curr = (j, w_dBm[j])
                # base = w_dBm[int(j/self.pps)]
                diff = curr[1] - base[1]
                # prev = curr
                if diff > w_trigger or diff < -w_trigger:
                    # Got an event @ index j
                    prev_time = curr_time
                    curr_time = self.timeline[i+j]
                    if curr_time - prev_time < 0.5:
                        # print("   ", curr_time, prev_time, curr_time-prev_time)
                        break
                    if diff > w_trigger:
                        a = "up"
                    else:
                        a = "down"
                    # print("@", self.timeline[i+base[0]], a, diff, w_trigger)
                    self.marks.append(self.timeline[i+base[0]])
                    duration, code, marks = self.decode_event(
                        i+j-int(self.pps/2))
                    gID = which_gesture(code)
                    print("@", self.timeline[i+base[0]],
                          code, gestureNames[gID])  # , self.timeline[i+duration])
                    # if (len(self.result) == 0) or (self.timeline[i+base[0]] - self.result[-1][0] > 3):
                    #     if gID != -1:
                    #         self.result.append(
                    #             (self.timeline[i+base[0]], code, gestureNames[gID]))
                    # if gestureNames[gID] != "No match":
                    #     self.marks += marks

                    # i += duration - 2*self.pps
                    break
                else:
                    base = (j, w_dBm[j])
        # for i in self.result:
        #     print("@", i[0], i[1], i[2])
        print("==========================")

    def decode_event(self, location):
        """ Decodes the event at location
        """
        marks = []
        sig_inc = 10

        # allowed max duration of gesture
        sig_offset = min(self.pps * 6, len(self.dBm[location:]))
        window = self.dBm[location: location+sig_offset]
        marks.append(self.timeline[location])
        marks.append(self.timeline[location+sig_offset-1])

        code_string = "n"
        curr = (0, window[0])
        trigger = (max(window) - min(window)) / 7
        # print(trigger)
        # timer = [0, 0, 0]
        ntimer = 0
        # print(self.timeline[location],
        #       self.timeline[location+sig_offset-1], sig_offset, trigger)
        for i in range(sig_inc, sig_offset, sig_inc):
            prev = curr
            curr = (i, window[i])
            diff = curr[1] - prev[1]
            slope = diff / \
                (self.timeline[curr[0]] - self.timeline[prev[0]])
            # print(diff, trigger)
            if slope > 0 and diff > trigger:
                # rising edge
                state = "r"
                # self.timer_set(0)
                ntimer = 0
            elif slope < 0 and diff < -trigger:
                # falling edge
                state = "f"
                # self.timer_set(1)
                ntimer = 0
            else:
                state = "n"
                # self.timer_set(2)
                ntimer += 1

            if state == code_string[-1]:
                continue
            else:
                # if self.timer_check():
                if state != "n" or ntimer > 10:
                    # self.timer_set(-1)
                    ntimer = 0
                    code_string += state
                    timestamp = self.timeline[location+prev[0]]
                    if timestamp not in self.marks:
                        marks.append(timestamp)

        # self.marks.append(self.timeline[location+i])
        return sig_offset, "s"+code_string[1:]+"e", marks

    def timer_check(self):
        rf_limit = 2
        n_limit = 10
        if self.timer[0] > rf_limit or self.timer[1] > rf_limit:
            return True

        if self.timer[1] > n_limit:
            return True
        return False

    def timer_set(self, sel):
        for i in range(0, 3):
            if i == sel:
                self.timer[i] += 1
            else:
                self.timer[i] = 0

    def plot_dBm(self):
        """ Plots out the dBm signal
        """
        plt.figure(figsize=(12, 2))
        plt.plot(self.timeline, self.dBm, markevery=[
            x in self.marks for x in self.timeline], marker="o")
        plt.xlabel('Time (s)')
        plt.ylabel('RSS (dBm)')
        # plt.savefig(filename[:-3]+".png")
        plt.show()


if __name__ == "__main__":
    n_start = 0
    n_tests = 9
    for i in range(n_start, n_tests+1):
        # i = 1
        filename = datapath + "down_hold_up/0{}.csv".format(i)
        print(filename)
        salut = RSSalut(filename)
