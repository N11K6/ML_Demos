#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:03:21 2021

@author: nk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
#%%
def generate_input(train_dir, batch_size, width, height):
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

    train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(height, width), su
            subset='training',
            batch_size=BATCH_SIZE,
            class_mode='categorical'
            )

    val_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(height, width), 
            subset='validation',
            batch_size=BATCH_SIZE,
            class_mode='categorical'
            )
    return train_generator, val_generator
#%%
def plot_examples(x_batch, y_batch):
    plt.figure(figsize=(12, 9))
    for k, (img, lbl) in enumerate(zip(x_batch, y_batch)):
        plt.subplot(4, 8, k+1)
        plt.imshow((img + 1) / 2)
        plt.title(int(lbl[1]))
        plt.axis('off')
#%%
def calculate_spe(y):
    return int(round((1. * y) / BATCH_SIZE))
#%%
def design_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(HEIGHT,WIDTH,3)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizers.Adam(lr=0.0001, epsilon=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    return model
#%%
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
  
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()
#%%
if __name__ == "__main__":
    WIDTH = 224
    HEIGHT = 224
    BATCH_SIZE = 24
    STEPS_PER_EPOCH = calculate_spe(1033*0.8)
    VALIDATION_STEPS = calculate_spe(1033*0.2)
    EPOCHS = 20

    traingen, valgen = generate_input(TRAIN_DIR, BATCH_SIZE, WIDTH, HEIGHT)
    
    model = design_model()
    
    history = model.fit(
            train_generator,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_data=val_generator,
            validation_steps=VALIDATION_STEPS,
            callbacks=my_callbacks)

    plot_training(history)
