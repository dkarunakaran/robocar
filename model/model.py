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
        df = pd.read_csv(self.PATH_TO_CSV, index_col=False)
        df.columns = ['Image', 'Steering', 'Throttle']
        df = df.sample(n=len(df))

        return df

    # Randomly selecting the let, right, and center images
    def random_select_image(self, data, i):
        path = data['Image'][i]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.RESIZE_SHAPE, interpolation=cv2.INTER_AREA)
        angle = float(data['Steering'][i])

        return image, angle

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
    
    def crop_image(self, image):
        """
        Returns an image cropped 40 pixels from top and 20 pixels from bottom.
        :param image: Image represented as a numpy array.
        """
        return image[50:,:]

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

    def gaussian_blur(self, image):
        gaussianbandwidths = [1, 3, 1, 1]
        gaussianidx = np.random.randint(0, len(gaussianbandwidths))
        gaussianbandwidth = gaussianbandwidths[gaussianidx]
        return cv2.GaussianBlur(image, (gaussianbandwidth,gaussianbandwidth), 0)

    # Getting fetatures and lables from training and validation data
    def get_data(self, data, training=False):
        images = []
        angles = []
        straight_count = 0
        not_considered_straight_count = 0
        for i in data.index:
            image, angle = self.random_select_image(data, i)
            #image = self.crop_image(image)
            if abs(angle) < .1:
                straight_count += 1
            if straight_count > (len(data) * .5):
                if abs(angle) < .1:
                    not_considered_straight_count +=1
                    continue

            images.append(image)
            angles.append(angle)

            if abs(angle) >= 0:
                image1, angle1 = self.flip_img_angle(image, angle)
                images.append(image1)
                angles.append(angle1)
            
            if random.randrange(3) == 1 and abs(angle) == 0:
                image1, angle1 = self.flip_img_angle(image, angle)
                images.append(image1)
                angles.append(angle1)

            if training == True:
                '''
                if random.randrange(3) == 1:
                    image1 = self.gaussian_blur(image)
                    images.append(image1)
                    angles.append(angle)'''
                image1 = self.brightnessed_img(image)
                images.append(image1)
                angles.append(angle)


        print("Toal files: "+str(len(data)))
        print("Total straight files: "+str(straight_count))
        print("Straight files not considered: "+str(not_considered_straight_count))
        
        angles = self.bin_matrix(angles)
        #angles = to_categorical(angles, 10)

        # Creating as numpy array
        X = np.array(images)
        y = np.array(angles)

        return X, y

    def normalize(self, image):
        """
        Returns a normalized image with feature values from -1.0 to 1.0.
        :param image: Image represented as a numpy array.
        """
        return image / 127.5 - 1.

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
samples = modelClass.get_csv()
samples = shuffle(samples, random_state=0)

# Training and Validation data
training_count = int(0.8 * len(samples))
training_data = samples[:training_count].reset_index()
validation_data = samples[training_count:].reset_index()

#data = samples.reset_index()
X_train, y_train = modelClass.get_data(training_data, training=True)
X_val, y_val = modelClass.get_data(validation_data, training=False)

print("Training sample: {}".format(len(X_train)))
print("Validation sample: {}".format(len(X_val)))

# Model using Keras
model = modelClass.get_model()

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', min_delta=.0001, patience=20, mode='auto'),
             ModelCheckpoint(filepath='./model.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min')]

#checkpoint = ModelCheckpoint("./model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#callbacks_list = [checkpoint]

# Training the model
#model.fit(X, y, shuffle=True, verbose=1, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=callbacks_list, epochs=EPOCH)
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, callbacks=callbacks, validation_data=(X_val, y_val), shuffle=True,initial_epoch=0)

