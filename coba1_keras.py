# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:04:10 2023

@author: admin
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import os

#%%

train_imgs = []
train_labels = []
path = "./dataset_train"
valid_images = [".jpg",".png"]
for file in os.listdir(path):
    ext = os.path.splitext(file)[1]
    class_img = file[:3]
    if ext.lower() not in valid_images:
        continue
    
    
    
    imagePath = path + '\\' + file
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    train_imgs.append(image)
    
    label = 0 if class_img == 'dog' else 1
    train_labels.append(label)


test_imgs = []
test_labels = []
path_test = "./dataset_test"
valid_images = [".jpg",".png"]
for file in os.listdir(path_test):
    ext = os.path.splitext(file)[1]
    class_img = file[:3]
    if ext.lower() not in valid_images:
        continue
    
    
    
    imagePath = path + '\\' + file
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    test_imgs.append(image)
    
    label = 0 if class_img == 'dog' else 1
    test_labels.append(label)
#%%    
train_imgs = np.array(train_imgs, dtype="float") / 255.0
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels)

test_imgs = np.array(test_imgs, dtype="float") / 255.0
test_labels = np.array(test_labels)
test_labels = to_categorical(test_labels)

#%%

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(2, activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 80
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#%%

# train the neural network
H = model.fit(x=train_imgs, y=train_labels, validation_data=(test_imgs, test_labels),
	epochs=EPOCHS, batch_size=2)

























