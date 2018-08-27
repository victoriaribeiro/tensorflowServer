import keras as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import itertools


def main():
    train_path = '/home/victoria/Downloads/Food-11/training'
    valid_path = '/home/victoria/Downloads/Food-11/validation'
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(255,255),  
    classes=['bread','dairy','dessert' , 'egg'  ,'fried',  'meat',  'pasta',  'rice',  'seafood',  'soup',  'vegetables'
], batch_size=20 )

    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(255,255),  
        classes=['bread','dairy','dessert' , 'egg'  ,'fried',  'meat',  'pasta',  'rice',  'seafood',  'soup',  'vegetables'
    ], batch_size=20 )

    model = Sequential([
        Conv2D(32,(3,3), activation='relu', input_shape=(255,255,3)),
        Flatten(),
        Dense(11,activation='softmax'),
    ])

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4,
    epochs=5,verbose=2)

main()