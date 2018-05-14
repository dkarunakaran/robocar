from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Conv2D, ELU, Flatten, Dense, Dropout, Activation, MaxPooling2D, Reshape, BatchNormalization, Input, Dense, merge
from keras.preprocessing.image import ImageDataGenerator, random_shift
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv

class ModelClass:

    # Paths
    PATH_TO_CSV = 'drive_data/training_data/drive_data.csv'

    # Shape
    INPUT_SHAPE = (240, 320, 3) #(Height, Width, Depth)
    RESIZE_SHAPE = (160,120) #(Width, Height)
    NEW_SHAPE = (120, 160, 3) #(Height, Width, Depth)
    LEARNING_PARAMETER = 0.0001 #.001

    # Get data from csv
    def get_csv(self):
        image_paths = []
        angles = []

        # Import driving data from csv
        with open(self.PATH_TO_CSV, newline='') as f:
            driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))
            data = []
            for row in driving_data[1:]:
                if len(row) > 0:
                    data.append(row)
            straight_count = 0
            not_considered_straight_count = 0
            for row in data:
                if abs(float(row[1])) < 0.01:
                    straight_count += 1
                if straight_count > (len(data) * .5):
                    if abs(float(row[1])) < 0.01:
                        not_considered_straight_count +=1
                        continue
                image_paths.append(row[0])
                angles.append(float(row[1]))
            print("Total files: {}".format(len(data)))
            print("Total straight count: {}".format(straight_count))
            print("Not considered stright count: {}".format(not_considered_straight_count))

        return image_paths, angles


    # Flipping the images
    def flip_img_angle(self, image, angle):
        image = cv2.flip(image, 1)
        angle *= -1.0

        return image, angle

    def linear_bin(self, value_to_bin):
        value_to_bin = value_to_bin + 1
        binned_value = round(value_to_bin / (2.0 / 14.0))

        return int(binned_value)

    def bin_matrix(self, matrix_to_bin):
        binned_matrix = []
        for value_to_bin in matrix_to_bin:
            temp_bin = np.zeros(15)
            temp_bin[self.linear_bin(value_to_bin)] = 1
            binned_matrix.append(temp_bin)
            
        return np.array(binned_matrix) 
    

    # Getting brightnessed image
    def brightnessed_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = .25 + np.random.uniform()
        #random_bright = .50 + np.random.uniform(low=0.0, high=.50)
        image[:,:,2] = image[:,:,2] * random_bright
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        #Convert to YUV color space (as nVidia paper suggests)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        return image

    def generate_training_data(self, image_paths, angles, batch_size=128, validation_flag=False):
        '''
        method for the model training data generator to load, process, and distort images, then yield them to the
        model. if 'validation_flag' is true the image is not distorted. also flips images with turning angle magnitudes of greater than 0.33, as to give more weight to them and mitigate bias toward low and zero turning angles
        '''
        image_paths, angles = shuffle(image_paths, angles)
        X,y = ([],[])
        while True:       
            for i in range(len(angles)):
                image = cv2.imread(image_paths[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
                angle = angles[i]
                if not validation_flag:
                    if random.randrange(3) == 1 and abs(angle) > 0:
                        image, angle = self.flip_img_angle(image, angle)
                    image = self.brightnessed_img(image)
                
                X.append(image)
                y.append(angle)
                
                if len(X) == batch_size:
                    y = self.bin_matrix(y)
                    yield (np.array(X), np.array(y))
                    X, y = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)

    # Creating the model
    def get_model(self):
        img_in = Input(shape=self.NEW_SHAPE, name='img_in')                      
        x = img_in
        x = Lambda(lambda x: x/255.-0.5,input_shape=self.NEW_SHAPE)(x)
        x = Cropping2D(cropping=((40, 0), (0, 0)))(x)
        x = Conv2D(24, (5,5), strides=(2,2), activation='relu')(x)       
        #x = Dropout(.1)(x)  
        x = Conv2D(32, (5,5), strides=(2,2), activation='relu')(x)       
        #x = Dropout(.1)(x)  
        x = Conv2D(64, (5,5), strides=(2,2), activation='relu')(x)       
        #x = Dropout(.1)(x)  
        x = Conv2D(64, (3,3), strides=(2,2), activation='relu')(x)       
        #x = Dropout(.1)(x)  
        x = Conv2D(64, (3,3), strides=(1,1), activation='relu')(x)      

        x = Flatten(name='flattened')(x)   
        #x = Dropout(.1)(x)  
        x = Dense(100, activation='relu')(x)     
        x = Dropout(.1)(x)                      
        x = Dense(50, activation='relu')(x)    
        x = Dropout(.1)(x)                     
        
        #categorical output of the angle
        angle_out = Dense(15, activation='softmax', name='angle_out')(x)        
        
        #continous output of throttle
        #throttle_out = Dense(1, activation='relu', name='throttle_out')(x)   
        
        model = Model(inputs=[img_in], outputs=[angle_out])
        #adam = Adam(lr=self.LEARNING_PARAMETER)
        rmsprop = RMSprop(lr=self.LEARNING_PARAMETER)
        model.compile(optimizer=rmsprop, loss={'angle_out': 'categorical_crossentropy'}, metrics=['accuracy'])

        return model


# Hyperparameteres
BATCH_SIZE = 128
EPOCH = 100

modelClass = ModelClass()

# Getting data from CSV
image_paths, angles = modelClass.get_csv()

# Training and Validation data
X_train, X_val, y_train, y_val = train_test_split(image_paths, angles, test_size=0.2, shuffle=True)

print("Training sample: {}".format(len(X_train)))
print("Validation sample: {}".format(len(X_val)))

# initialize generators
train_gen = modelClass.generate_training_data(X_train, y_train, validation_flag=False, batch_size=BATCH_SIZE)
val_gen = modelClass.generate_training_data(X_val, y_val, validation_flag=True, batch_size=BATCH_SIZE)

steps_per_epoch = int(len(X_train)/BATCH_SIZE)
validation_steps = int(len(X_val)/BATCH_SIZE)

# Model using Keras
model = modelClass.get_model()

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', min_delta=.0001, patience=25, mode='auto'),
             ModelCheckpoint(filepath='./model.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')]

# Training the model
#model.fit(X, y, shuffle=True, verbose=1, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks_list, epochs=EPOCH)
model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCH, verbose=1, callbacks=callbacks, validation_data=val_gen, validation_steps=validation_steps)

