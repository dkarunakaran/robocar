#!/usr/bin/env python

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


csv_lines = []
csv_drive_data = 'drive_data/training_data/drive_data.csv'
with open(csv_drive_data) as csvfile:
	reader = csv.reader(csvfile, delimiter=",", quotechar='|')
	for line in reader:
		csv_lines.append(line)

print(csv_lines)
steering_commands = []

left_count = 0
right_count = 0
zero_count = 0
count = 0
for csv_line in csv_lines:
    if count != 0:
        steering_angle = float(csv_line[1])
        
        if steering_angle > 0:
            left_count += 1
        elif steering_angle < 0:
            right_count += 1
        else:
            zero_count +=1
    count += 1

print("Left turns count: " + str(left_count))
print("Right turns count: " + str(right_count))
print("Zero count: " + str(zero_count))
