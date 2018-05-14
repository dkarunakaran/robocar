#!/usr/bin/env python

import os
import csv
import cv2
import numpy as np 

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
                    image_name = row[0].split('.')
                    f = open(file_path+image_name[0], 'rb')
                    image_data = f.read()
                    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (320,240), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(image_path, image)
                    steering = row[1]
                    throttle = row[2]
                    filewriterfiles.writerow([image_path, steering, throttle])
                    print("{} {} {}".format(image_path, steering, throttle))
                    print("")
                count +=1

            





