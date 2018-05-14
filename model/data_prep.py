#!/usr/bin/env python

import os
import csv
for root, directories, filenames in os.walk('drive_data/training_data'):
    with open(root+"/"+"drive_data.csv", 'w') as drivecsv:
        filewriterfiles = csv.writer(drivecsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        filewriterfiles.writerow(['Image', 'Steering', 'Throttle'])
        for directory in directories:
            file_path = root+"/"+directory+"/"
            data_file_path  = file_path+'data.csv'
            f = open(data_file_path, 'r')
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if count != 0:
                    image_path = file_path+row[0]
                    steering = row[1]
                    throttle = row[2]
                    filewriterfiles.writerow([image_path, steering, throttle])
                    print("{} {} {}".format(image_path, steering, throttle))
                    print("")
                count +=1
