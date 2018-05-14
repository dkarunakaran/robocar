#!/usr/bin/env python

import time
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import CompressedImage
#import numpy as np
import os
import csv

class TrainingNode:

    steering_pulse = None
    throttle_pulse = None

    #Steering constants
    MAX_LEFT_ANGLE = rospy.get_param("/MAX_LEFT_ANGLE")
    MIN_RIGHT_ANGLE = rospy.get_param("/MIN_RIGHT_ANGLE")
    MAX_STEERING_PULSE = rospy.get_param("/MAX_STEERING_PULSE")
    MIN_STEERING_PULSE = rospy.get_param("/MIN_STEERING_PULSE")
    START_RECORDING  = None

    #Throttle constants
    MIN_THROTTLE = rospy.get_param("/MIN_THROTTLE")
    MAX_THROTTLE = rospy.get_param("/MAX_THROTTLE")
    MIN_THROTTLE_PULSE = rospy.get_param("/MIN_THROTTLE_PULSE")
    MAX_THROTTLE_PULSE = rospy.get_param("/MAX_THROTTLE_PULSE")
    TRAINING_DATA_DIRECTORY = '/home/dhanoop/catkin_ws/training_data'

    folder_name = None
    data_file = None
    data = []
    folder_location = None
    log = []
    log_file = []

    def __init__(self):
        self.folder_name = str(int(round(time.time() * 1000)))
        if not os.path.isdir(self.TRAINING_DATA_DIRECTORY):
            os.makedirs(self.TRAINING_DATA_DIRECTORY)
        if not os.path.isdir(self.TRAINING_DATA_DIRECTORY+"/"+self.folder_name):
                os.makedirs(self.TRAINING_DATA_DIRECTORY+"/"+self.folder_name)
        self.folder_location = self.TRAINING_DATA_DIRECTORY+"/"+self.folder_name
        self.data_file = open(self.TRAINING_DATA_DIRECTORY+'/'+self.folder_name+'/'+'data.csv', 'w')
        self.log_file = open('log.csv', 'w')


    def camera_node(self, image):
        self.log.append(['camera starts: '+str(time.time() * 1000)])
        self.START_RECORDING  = rospy.get_param("/START_RECORDING")
        if self.START_RECORDING == True:
            millis = int(round(time.time() * 1000))
            image_name = "".join(['image_',str(millis)])
            full_path_name = ''.join([self.folder_location,'/',image_name])
            file = open(full_path_name,"wb")
            file.write(image.data) 
            file.close() 
            self.data.append([image_name+'.jpg',self.steering_pulse, self.throttle_pulse])
        self.log.append(['camera ends: '+ str(time.time() * 1000)])
        rospy.loginfo('Recording: %s', self.START_RECORDING)

    def joystick_node(self, data):
        self.log.append(['joystic starts: '+ str(time.time() * 1000)])
        self.steering_pulse = data.axes[0]
        self.throttle_pulse = self.find_max_throttle(data.axes[3])
        self.log.append(['joystic ends: '+ str(time.time() * 1000)])

    def find_max_throttle(self, throttle_pulse):
        throttle = rospy.get_param("/CONST_THROTTLE")
        if throttle_pulse <= throttle:
            throttle = throttle_pulse

        return throttle
    
    def shutdown(self):
        with self.data_file:
            writer = csv.writer(self.data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Image','Steering', 'Throttle'])
            writer.writerows(self.data)

        with self.log_file:
            writer = csv.writer(self.log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Log'])
            writer.writerows(self.log)
        rospy.loginfo('Saving the data')


def listener(trainingNode):

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('training_node', anonymous=True)
    rospy.Subscriber('raspicam_node/image/compressed', CompressedImage, trainingNode.camera_node, queue_size=30)
    rospy.Subscriber("joy", Joy, trainingNode.joystick_node, queue_size=30)
    rospy.on_shutdown(trainingNode.shutdown)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    trainingNode = TrainingNode()
    listener(trainingNode)
