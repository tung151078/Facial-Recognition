import os, cv2, PIL, imageio
import pathlib
import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet_v2 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
%matplotlib inline

data_dir = pathlib.Path('/content/Data')

# Save result on google drive
PROJECT = data_dir
RESULT = PROJECT/'Results'
SAVED_MODEL = RESULT/'Saved_model'
IMG_SIZE = 224 
BATCH_SIZE = 32

# Prepare ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generator_augmentation_maker():
    train_gen = ImageDataGenerator(
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest',
                                    validation_split=0.2)     
    
    val_gen = ImageDataGenerator(validation_split=0.2)  

    train_generator = train_gen.flow_from_directory(
                                    data_dir,  
                                    target_size=(IMG_SIZE, IMG_SIZE),  
                                    batch_size=BATCH_SIZE,
                                    class_mode='sparse',
                                    shuffle=True,
                                    seed=42,           
                                    subset='training')  

    validation_generator = val_gen.flow_from_directory(
                                    data_dir,
                                    target_size=(IMG_SIZE, IMG_SIZE),
                                    batch_size=BATCH_SIZE,
                                    class_mode='sparse',
                                    shuffle=False,       
                                    seed=42,              
                                    subset='validation')  
    return train_generator, validation_generator

train_generator, validation_generator = generator_augmentation_maker()

# Build model
base_model = keras.applications.MobileNetV2(weights='imagenet',
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   include_top=False)
# Freeze the base model's weights
base_model.trainable=False

def my_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE,3))
    x = preprocess_input(inputs)

    x = base_model(x,training=False)

    x = layers.Dense(256)(x)
    x = layers.Dense(512)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model

model = my_model()

model.compile(optimizer= 'Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ear_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
mod_chk = ModelCheckpoint(filepath='/content/gdrive/MyDrive/Project_ML30/face_detection/face_detect.h5', monitor='val_loss', save_best_only=True)
lr_rate = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)

hist = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=20,
                    callbacks = [ear_stop, mod_chk, lr_rate], verbose=2)
# Save model
model.save('/content/gdrive/MyDrive/Project_ML30/face_detection/face_detect.h5') 