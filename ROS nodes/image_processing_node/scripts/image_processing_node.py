#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import tensorflow as tf
import h5py
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from keras.models import load_model
from random import uniform
import sys

class ImageProcessingNode:

    IMAGE_SCALE = 4
    model = None
    autodrive_publisher = None
    graph = None
    terminate = False

    def __init__(self):
        self.model = load_model("model.h5")
        self.graph = tf.get_default_graph()
        self.terminate = False

    def linear_unbin1(self, arr):
        b = np.argmax(arr)
        a = b *(2/14) - 1
        return a

    def linear_unbin(self, value_to_unbin):
        unbinned_value = value_to_unbin * (2.0 / 14.0) - 1

        return unbinned_value
    
    def unbin_matrix(self, matrix_to_unbin):
        unbinned_matrix=[]
        
        for value_to_unbin in matrix_to_unbin:
            unbinned_value = np.argmax(value_to_unbin)
            unbinned_value = self.linear_unbin(unbinned_value)
            unbinned_matrix.append(unbinned_value)

        return np.array(unbinned_matrix)

    def image_process(self, message):
        np_arr = np.fromstring(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (160,120), interpolation=cv2.INTER_AREA)
        #image = image[120:,:]
        image = np.asarray(image)
        images = []
        images.append(image)
        with self.graph.as_default():
            prediction = self.model.predict(np.array(images), batch_size=1)
            #steering_angle = prediction[0][0]
            steering_angle = self.linear_unbin1(prediction)
            throttle = rospy.get_param("/THROTTLE_AUTO_DRIVE")
            #throttle = prediction[1][0][0]
            command = str(steering_angle) + ":" + str(throttle)
            if self.terminate == False:
                rospy.loginfo("Autopilot command: " + command)
                self.autodrive_publisher.publish(command)

    def shutdown(self):
        self.terminate = True
        rospy.loginfo("Setting steering angle and throttle to 0")
        self.autodrive_publisher.publish("0:0")
        self.autodrive_publisher.publish("0:0")
        self.autodrive_publisher.publish("0:0")


def listener(imageProcessingNode):

    rospy.init_node("self_driving_node", log_level=rospy.DEBUG)
    imageProcessingNode.autodrive_publisher = rospy.Publisher("autodrive", String, queue_size=1)
    
    #initialising
    rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, imageProcessingNode.image_process, queue_size=1)
    rospy.on_shutdown(imageProcessingNode.shutdown)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    imageProcessingNode = ImageProcessingNode()
    listener(imageProcessingNode)