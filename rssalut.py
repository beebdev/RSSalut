#! /usr/bin/python3
import csv

with open("data/gesture_0/out1.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        print(row)
