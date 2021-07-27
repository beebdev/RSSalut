#! /usr/bin/python3

import pywt

fam = pywt.families()
for f in fam:
    print(pywt.wavelist(f))
