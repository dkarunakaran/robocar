#!/usr/bin/env python

import cv2
import numpy as np
import tensorflow as tf
import h5py
from keras.models import load_model
from random import uniform
import sys
import os
from keras.utils import to_categorical

def linear_unbin(value_to_unbin):
    unbinned_value = value_to_unbin * (2.0 / 14.0) - 1

    return unbinned_value
    
def unbin_matrix(matrix_to_unbin):
    unbinned_matrix=[]
    
    for value_to_unbin in matrix_to_unbin:
        unbinned_value = np.argmax(value_to_unbin)
        unbinned_value = linear_unbin(unbinned_value)
        unbinned_matrix.append(unbinned_value)

    return np.array(unbinned_matrix)

def linear_unbin1(arr):
    b = np.argmax(arr)
    a = b *(2/14) - 1
    return a


def normalize(image):
    """
    Returns a normalized image with feature values from -1.0 to 1.0.
    :param image: Image represented as a numpy array.
    """
    return image / 127.5 - 1.


'''
images = []
model = load_model("model.h5")
print("image_1518037766438")
last_image1 = cv2.imread('image_1518037766438.jpg',cv2.IMREAD_COLOR)
last_image1 = cv2.resize(last_image1, (160, 120), interpolation=cv2.INTER_AREA)
images.append(np.asarray(last_image1))
value = model.predict(np.array(images), batch_size=1)
print(value)
print(unbin_matrix(value)[0])
images = []
print("image_1518074268596")
last_image2 = cv2.imread('image_1518074268596.jpg',cv2.IMREAD_COLOR)
last_image2 = cv2.resize(last_image2, (160,120), interpolation=cv2.INTER_AREA)
images.append(np.asarray(last_image2))
value = model.predict(np.array(images), batch_size=1)
print(value)
print(unbin_matrix(value)[0])
images = []
print("image_1518507584011")
last_image3 = cv2.imread('image_1518507584011.jpg',cv2.IMREAD_COLOR)
#last_image3 = last_image3[120:,:]
last_image3 = cv2.resize(last_image3, (160,120), interpolation=cv2.INTER_AREA)
images.append(np.asarray(last_image3))
value = model.predict(np.array(images), batch_size=1)
print(value)
print(unbin_matrix(value)[0])
images = []
print("image_1518841302776")
last_image4 = cv2.imread('image_1518841302776.jpg',cv2.IMREAD_COLOR)
last_image4 = cv2.resize(last_image4, (160,120), interpolation=cv2.INTER_AREA)
images.append(np.asarray(last_image4))
value = model.predict(np.array(images), batch_size=1)
print(value)
print(unbin_matrix(value)[0])
#last_image5 = cv2.imread('image_1518507584011.jpg',cv2.IMREAD_COLOR)
#last_image5 = last_image5[120:,:]
#last_image5 = cv2.resize(last_image5, (320,120), interpolation=cv2.INTER_AREA)
#images.append(np.asarray(last_image5))

#last_image6 = cv2.imread('image_1518507587784.jpg',cv2.IMREAD_COLOR)
#last_image6 = last_image6[120:,:]
#images.append(np.asarray(last_image6))

#value = model.predict(np.array(images), batch_size=1)
#steering_angle = linear_unbin(value)
#print(steering_angle)
#print(value)

'''
model = load_model("model.h5")
path = 'test_model_images1'
for filename in os.listdir(path):
    print(filename)
    images = []
    last_image1 = cv2.imread(path+'/'+filename,cv2.IMREAD_COLOR)
    last_image1 = cv2.resize(last_image1, (160, 120), interpolation=cv2.INTER_AREA)
    images.append(np.asarray(last_image1))
    value = model.predict(np.array(images), batch_size=1)
    #print(unbin_matrix(value)[0])
    print(linear_unbin1(value))
    #print(value.argmax(1))

print("_______________________________________________")

path = 'test_model_images2'
for filename in os.listdir(path):
    print(filename)
    images = []
    last_image1 = cv2.imread(path+'/'+filename,cv2.IMREAD_COLOR)
    last_image1 = cv2.resize(last_image1, (160, 120), interpolation=cv2.INTER_AREA)
    images.append(np.asarray(last_image1))
    value = model.predict(np.array(images), batch_size=1)
    #print(unbin_matrix(value)[0])
    print(linear_unbin1(value))
    #print(value.argmax(1))

print("_______________________________________________")

path = 'test_model_images3'
for filename in os.listdir(path):
    print(filename)
    images = []
    last_image1 = cv2.imread(path+'/'+filename,cv2.IMREAD_COLOR)
    last_image1 = cv2.resize(last_image1, (160, 120), interpolation=cv2.INTER_AREA)
    images.append(np.asarray(last_image1))
    value = model.predict(np.array(images), batch_size=1)
    print(linear_unbin1(value))
    #print(value)
    #print(value.argmax(1))