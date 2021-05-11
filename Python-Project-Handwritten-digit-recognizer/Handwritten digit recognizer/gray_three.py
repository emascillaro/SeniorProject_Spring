import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image
from PIL import Image, ImageOps, ImageChops

CATEGORIES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
PATH = r"jpg_photos"

'''
# (SECOND BEST PREPARE METHOD TO USE)
# Formatting image for model 
def prepare(path):      
    try:
        IMG_SIZE = 28
        img = Image.open(path)
        img = ImageOps.grayscale(img)
      # img = ImageChops.invert(img)
        print('Orignal Image:', img.size)
        resized_img = img.resize((IMG_SIZE, IMG_SIZE))
        resized_img.save('66_resized.jpg', quality=95)                             # save the image as something else, not the name of the original photo
        resized_img.show()
        print('Resized Grayscale Image:', resized_img.size)
        img_array = cv2.imread('66_resized.jpg')
        new_array = img_array.astype('float32')
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        return new_array
    except Exception as e:
        print(str(e))


model = tf.keras.models.load_model('gray_model.h5', compile = True)
os.chdir("jpg_photos")
os.system("cd")
prediction = model.predict([prepare('66.jpg')])                           #always pass a list
print(CATEGORIES[int(prediction[0][0])])

'''
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
os.chdir("jpg_photos")
os.system("cd")
prediction = model.predict([prepare('66.jpg')])                           #always pass a list
# print(np.argmax(prediction))
pred_name = CATEGORIES[np.argmax(prediction)]
print("Model's Prediction:", pred_name)

# Place this code before speciying prediction and after os.system
'''
img = Image.open('50.jpg')
inv_img = ImageChops.invert(img)
inv_img.save('inv_50.jpg')
inv_img.show()
'''


# OLDER PREPARE METHODS AND CODE

'''
def prepare(path):      
    try:
        IMG_SIZE = 28
        img = Image.open(path)
        img = ImageOps.grayscale(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img.save('8_gray.jpg')
        img.show()
        print(img.size)
        img_array = cv2.imread('8_gray.jpg')
        new_array = img_array.astype('float32')
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #new_array = new_array / 255.0
        #print(new_array)
        return new_array
    except Exception as e:
        print(str(e))



def prepare(path):
    try: 
        IMG_SIZE = 28
       # path = os.path.join(path, img)
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
       # plt.imshow(img_array, cmap = plt.cm.gray)
        #plt.show()
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        plt.imshow(new_array, cmap = plt.cm.gray)
        plt.show()
        new_array = new_array.astype('float32')
        new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        return new_array
    except Exception as e:
        print(str(e))

model = tf.keras.models.load_model('gray_model.h5', compile = True)
#model.compile(loss="sparse_categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])
#os.system("cd jpg_photos")
os.chdir("jpg_photos")
os.system("cd")
prediction = model.predict([prepare('8.jpg')])      #always pass a list
#print(prediction)   # Will be a list in a list
print(CATEGORIES[int(prediction[0][0])])

#classes = np.argmax(prediction, axis = 1)
#print(classes)        


model = tf.keras.models.load_model('new_model.h5')
model.compile(loss="sparse_categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])
IMG_SIZE = 28
img = image.load_img(image_path, (IMG_SIZE,IMG_SIZE))
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
plt.imshow(img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print(CATEGORIES[int(prediction[0][0])])


'''
