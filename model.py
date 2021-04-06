# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 04:23:43 2021

@author: Shrita
"""

import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import random

img_array = cv2.imread('train/Closed_Eyes/s0001_00001_0_0_0_0_0_01.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array, cmap="gray")


Datadirectory = 'train'
Classes = ['Closed_Eyes', 'Open_Eyes']
for category in Classes:
    path = os.path.join(Datadirectory, category)     
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

img_size = 224
new_array = cv2.resize(backtorgb, (img_size,img_size))
plt.imshow(new_array, cmap="gray")
plt.show()

training_data = []

def create_training_data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb, (img_size,img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
            
create_training_data()

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, img_size, img_size, 3)

X = X/255.0

Y = np.array(y)


model = tf.keras.applications.mobilenet.MobileNet()

base_input = model.layers[0].input
base_output = model.layers[-4].output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)

new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

new_model.fit(X,Y, epochs = 2, validation_split = 0.1)
new_model.save('my_model.h5')



#prediction on new data

new_model = tf.keras.models.load_model('my_model.h5')


img_array3 = cv2.imread('train/Open_Eyes/s0001_02358_0_0_1_0_0_01.png',cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array3, cv2.COLOR_GRAY2BGR)
new_array = cv2.resize(backtorgb, (img_size, img_size))

X_input = np.array(new_array).reshape(1, img_size, img_size, 3)
print(X_input.shape)
X_input = X_input/255.00 
prediction = new_model.predict(X_input)

print(prediction)


























