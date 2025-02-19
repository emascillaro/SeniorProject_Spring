import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image
from PIL import Image, ImageOps, ImageChops

import bound_box

CATEGORIES_digit = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
CATEGORIES_operator = ["+", "-", "*", "/"]

# Calling bound_box function 
box_heights = bound_box.box()
prepared_img_list = []
final_predict_string = ""
#final_predict_list = []

# (BEST PREPARE METHOD TO USE)
# Formatting image for model 
def prepare(path):      
    try:
        IMG_SIZE = 28
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print('Orignal Image Array Shape:', img_array.shape)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = new_array.astype('float32')
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        print('Resized Grayscale Image Array for Model:', new_array.shape)
        return new_array
    except Exception as e:
        print(str(e))

# Compile model
model_digit = tf.keras.models.load_model('gray_model.h5', compile = True)
model_operator = tf.keras.models.load_model('gray_operators_model.h5', compile = True)  

# Switch directory to folder with boxed images from box()
directory = r'final_images'

# Iterate through images in folder and convert to arrays using prepare()
# Add image arrays to empty list
count = 1
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f) and filename.endswith('.jpg'): 
        print("Image #", count)
        count += 1
        prepared_image = prepare(f)
        prepared_img_list.append(prepared_image)

# Iterate through ordered image array list (left to right just like we read math) and predict each image
# Depending on height of orignal boxed image, we either use digit model or operator model 
# Add image prediction to another empty list 
h_count = 0
for i in prepared_img_list:
    if box_heights[h_count] <= 45:
        prediction = model_operator.predict(i) 
        pred_name = CATEGORIES_operator[np.argmax(prediction)]
        final_predict_string += pred_name
        #final_predict_list.append(pred_name)
        print("Model's Prediction:", pred_name)
        h_count += 1
    else:
        prediction = model_digit.predict(i) 
        pred_name = CATEGORIES_digit[np.argmax(prediction)]
        final_predict_string += pred_name
        #final_predict_list.append(pred_name)
        print("Model's Prediction:", pred_name)
        h_count += 1

print(final_predict_string)

import solution_window

'''


print(final_predict_list)
#print(final_predict_string)
#numeric(final_predict_string)

# Remove ALL image files from folder 
for filename in os.listdir(directory):
   os.remove(os.path.join(directory, filename))

'''

'''
# Convert String List to Integer List               #SOMETHING IS REALLY WRONG WITH THIS
for i in final_predict_list:
    if i == "+":
        final_predict_list.append(i)
    else:
        final_predict_list = [int(i) for i in final_predict_list]
print(final_predict_list)
'''



'''
CATEGORIES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
PATH = r"final_images"

# (BEST PREPARE METHOD TO USE)
# Formatting image for model 
def prepare(path):      
    try:
        IMG_SIZE = 28
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print('Orignal Image Array Shape:', img_array.shape)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array = new_array.astype('float32')
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        print('Resized Grayscale Image Array for Model:', new_array.shape)
        return new_array
    except Exception as e:
        print(str(e))


model = tf.keras.models.load_model('gray_model.h5', compile = True)
os.chdir("final_images")
os.system("cd")
prediction = model.predict([prepare('ROI_2.jpg')])                           #always pass a list
# print(np.argmax(prediction))
pred_name = CATEGORIES[np.argmax(prediction)]
print("Model's Prediction:", pred_name) 

'''

