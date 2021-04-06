# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 01:50:06 2021

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
import io
import random
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from base64 import b64encode


app = Flask(__name__)


@app.route('/')

def home():
    return render_template('driver.html')


@app.route('/predicteyes',methods = ['POST'])
def predicteyes():
    
    
    photo = request.files['file']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = cv2.IMREAD_GRAYSCALE
    img_array3 = cv2.imdecode(data, color_image_flag)
    
    #file = b64encode(file) 
     
    
 
    img_size = 224
    
    new_model = tf.keras.models.load_model('my_model.h5')
    

      
    #img_array3 = cv2.imread('file',cv2.IMREAD_GRAYSCALE)

    backtorgb = cv2.cvtColor(img_array3, cv2.COLOR_GRAY2BGR)
    new_array = cv2.resize(backtorgb, (img_size, img_size))
     
    X_input = np.array(new_array).reshape(1, img_size, img_size, 3)
    print(X_input.shape)
    X_input = X_input/255.00 
    prediction = new_model.predict(X_input)
    if prediction > 0.5:
        prediction_text = "The image that you entered has open eyes"
    elif prediction < 0.5:
        prediction_text = "The image that you entered has closed eyes"
    else:
        prediction_text = "undetermined"
        

    
    return render_template('driver.html',prediction_text2 = 'Prediction: {}'.format(prediction_text))

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
