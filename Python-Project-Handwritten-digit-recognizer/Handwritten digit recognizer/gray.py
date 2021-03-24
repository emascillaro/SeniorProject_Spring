import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random 
import pickle
from PIL import ImageOps

# Directories and Categories for MNIST jpg dataset 
DATADIR = r"MNIST_Dataset_JPG_format\MNIST_JPG_training"
DATADIR_TEST = r"MNIST_Dataset_JPG_format\MNIST_JPG_testing"
CATEGORIES = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

'''
for category in CATEGORIES:
  path = os.path.join(DATADIR, category)  # path to training data
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img))
    plt.imshow(img_array)
    plt.show()
    break
  break'''

# Empty lists for training and testing data
training_data = []
testing_data = []

def create_training_data():
    for category in CATEGORIES:
        path_train = os.path.join(DATADIR, category)  # path to training data
        class_num = CATEGORIES.index(category)  
        for img in os.listdir(path_train):      #iterate through images in their path, convert to grayscale
          try:
            img_array = cv2.imread(os.path.join(path_train,img), cv2.IMREAD_GRAYSCALE) # array of pixel values frm image
            training_data.append([img_array, class_num]) # adds image to training list
          except Exception as e:
            pass

def create_testing_data():
    for category in CATEGORIES:
        path_test = os.path.join(DATADIR_TEST, category)  # path to training data
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path_test):
          try:
            img_array = cv2.imread(os.path.join(path_test,img), cv2.IMREAD_GRAYSCALE)
            testing_data.append([img_array, class_num])
          except Exception as e:
            pass

# Printing the number of images in training and testing data
create_training_data()
print("Total Training Images:", len(training_data))
create_testing_data()
print("Total Testing Images:", len(testing_data))

#Shuffling training and testing data for proper balance since we iterated over categories
random.shuffle(training_data)
for sample in training_data[:10]:
  print(sample[1])

random.shuffle(testing_data)
for sample in testing_data[:10]:
  print(sample[1])

# Empty training and testing images and classes
X_train = []
y_train = []
X_test = []
y_test = []
#y = np.array(y)

# Building image and class lists
for features, label in training_data:
  X_train.append(features)
  y_train.append(label)
  #np.array((y,label))

for features, label in testing_data:
  X_test.append(features)
  y_test.append(label)
  #np.array((y,label))

# Convert lists into arrays for model
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Training Data Saved and Loaded
pickle_out = open("X_train.pickle", "wb")   
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_in = open("X_train.pickle", "rb")
X_train = pickle.load(pickle_in)

# Testing Data Saved and Loaded
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_in = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_in)

print("Training and Testing Datasets were successfully saved and loaded")

