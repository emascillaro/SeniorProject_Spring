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


X_train_operator = pickle.load(open("X_train_operator.pickle", "rb"))
y_train_operator = pickle.load(open("y_train_operator.pickle", "rb"))
X_test_operator = pickle.load(open("X_test_operator.pickle", "rb"))
y_test_operator = pickle.load(open("y_test_operator.pickle", "rb"))

X_train_operator = X_train_operator.reshape(X_train_operator.shape[0], 28, 28, 1)
X_test_operator = X_test_operator.reshape(X_test_operator.shape[0], 28, 28, 1)
y_train_operator = y_train_operator.reshape(-1,1)
y_test_operator = y_test_operator.reshape(-1,1)

print("X_train_operator shape:", X_train_operator.shape)
print("y_train_operator shape:", y_train_operator.shape)
print("X_test_operator shape:", X_test_operator.shape)
print("y_test_operator shape:", y_test_operator.shape)

# Normalizing the data
X_train_operator=X_train_operator/255.0
X_test_operator=X_test_operator/255.0

# OUR model
batch_size = 128
num_classes = 4
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

hist = model.fit(X_train_operator, y_train_operator, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test_operator, y_test_operator))
print("The model has successfully trained")

score = model.evaluate(X_test_operator, y_test_operator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# prediction = model.predict(X_test_operator) 
# np.argmax(prediction, axis=1)
# prediction.reshape(-1,1)
# print(prediction.shape)

# Print the precision and recall, among other metrics
# precision_score(y_test_operator, prediction, average = 'weighted')

model.save('gray_operators_model.h5')
print("Model is saved as gray_operators_model.h5")

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