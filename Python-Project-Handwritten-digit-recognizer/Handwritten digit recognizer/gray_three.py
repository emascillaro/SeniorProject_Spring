import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image
from PIL import Image, ImageOps

CATEGORIES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
PATH = r"jpg_photos"
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
'''


#Works kinda prints one every time; makes sense since we resized the image twice and it looks like one 
def prepare(path):      
    try:
        IMG_SIZE = 28
        img = Image.open(path)
        img = ImageOps.grayscale(img)
        print(img.size)
        resized_img = img.resize((IMG_SIZE, IMG_SIZE))
        resized_img.save('8_resized.jpg')        # save the image as something else, not the name of the original photo
        resized_img.show()
        print(resized_img.size)
        img_array = cv2.imread('8_resized.jpg')
        #plt.imshow(img_array, cmap = plt.cm.gray)
        #plt.show()
        #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        #plt.imshow(new_array, cmap = plt.cm.gray)
        #plt.show()
        new_array = img_array.astype('float32')
        new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        #new_array = new_array / 255.0
        #print(new_array)
        return new_array
    except Exception as e:
        print(str(e))


'''
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

        '''
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

'''
model = tf.keras.models.load_model('new_model.h5')
model.compile(loss="sparse_categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])

IMG_SIZE = 28
img = image.load_img(image_path, (IMG_SIZE,IMG_SIZE))
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
plt.imshow(img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print(CATEGORIES[int(prediction[0][0])])'''