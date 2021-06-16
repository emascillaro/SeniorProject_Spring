import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import PIL

DATADIR = r"MNIST_Dataset_JPG_format\Operators"
CATEGORIES = ["Minus_three"]   # names of photo files

total_array = []

dir_plus = r"MNIST_Dataset_JPG_format\Operators\resized_plus_three"
dir_minus = r"MNIST_Dataset_JPG_format\Operators\resized_minus_three"
dir_times = r"MNIST_Dataset_JPG_format\Operators\resized_times_three"
dir_divide = r"MNIST_Dataset_JPG_format\Operators\resized_divide_two"
count = 0

for category in CATEGORIES:
  path = os.path.join(DATADIR, category)       # path to operator imaegs
  for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        try:
            img_array = cv2.imread(os.path.join(path,filename))
            new_array = cv2.resize(img_array, (28, 28))
            total_array.append(new_array)
        except Exception as e:
            print(str(e))

os.chdir(dir_minus)         # can't be in a loop or else it gets confused. changing directory to the same directory? doesn't work

for i in total_array:
    image = Image.fromarray(i)
    image.save('test_{}.jpg'.format(count))
    count += 1    



'''
for category in CATEGORIES:
  path = os.path.join(DATADIR, category)  # path to training data
  for filename in os.listdir(path):
    if filename.endswith(".jpg"):
        try:
            img_array = cv2.imread(os.path.join(path,filename))
            new_array = cv2.resize(img_array, (28, 28))
            #total_array.append(new_array)
            #plt.imshow(new_array)
            #plt.show()
            image = Image.fromarray(new_array)
            #print(image.shape)
            os.chdir(dir)
            image.save('test_{}.jpg'.format(count))
            count += 1
        except Exception as e:
            print(str(e))
    else:
        break
'''
   