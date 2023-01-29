#!/usr/bin/env python
import numpy as np
import csv

diffs = []
labels = np.empty((0,6), dtype=float)

with open('stats.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 1
    for row in csv_reader:
        diff = float(row[1]) - float(row[3])
        diffs.append(diff)
        
        label = np.array([[line_count, diff, row[0], row[1], row[2], row[3]]])
        labels = np.append(labels, label, axis=0)
        line_count += 1

for item in labels:
    print (item)
indices  = (-np.array(diffs)).argsort()

print("best diffs")

for i in range(20):
    print (labels[indices[i]])
