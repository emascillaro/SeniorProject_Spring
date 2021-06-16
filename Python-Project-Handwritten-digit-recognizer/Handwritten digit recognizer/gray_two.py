import tensorflow as tf;
import numpy as np
import pickle
import random
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import precision_score 


X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Normalizing the data
X_train=X_train/255.0
X_test=X_test/255.0

# OUR model
batch_size = 128
num_classes = 10
epochs = 5

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),input_shape=(28,28,1)))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer= 'adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
print("The model has successfully trained")

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# prediction = model.predict(X_test) 
# np.argmax(prediction, axis=1)
# prediction.reshape(-1,1)
# print(prediction.shape)

# Print the precision and recall, among other metrics
# precision_score(y_test, prediction, average = 'weighted')

model.save('gray_model.h5')
print("Model is saved as gray_model.h5")

'''
#Online Model

img_width, img_height = 28, 28
batch_size = 250
epochs = 5
no_classes = 10
validation_split = 0.2
verbosity = 1
leaky_relu_alpha = 0.1

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1)))
model.add(LeakyReLU(alpha=leaky_relu_alpha))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(LeakyReLU(alpha=leaky_relu_alpha))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=leaky_relu_alpha))
model.add(Dense(no_classes, activation='softmax'))'''
