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
import sys
from PIL import Image

def main():
    train_path = '/home/douglas/nutripic/food-11/training'
    valid_path = '/home/douglas/nutripic/food-11/validation'
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224),  
    classes=['bread','dairy','dessert' , 'egg'  ,'fried',  'meat',  'pasta',  'rice',  'seafood',  'soup',  'vegetables'
    ], batch_size=70 )

    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),  
        classes=['bread','dairy','dessert' , 'egg'  ,'fried',  'meat',  'pasta',  'rice',  'seafood',  'soup',  'vegetables'
    ], batch_size=70 )


    vgg16_model = keras.applications.vgg16.VGG16()

    type(vgg16_model)
    model = Sequential()
    i=0;
    for layer in vgg16_model.layers:
        i+=1
        if(i<=22):
            model.add(layer)

    model.summary()
    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False

    model.summary()

    model.add(Dense(11, activation="softmax"))
    model.summary()

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_batches, steps_per_epoch=480, validation_data=valid_batches, validation_steps=480, epochs=50,verbose=2)

    model.save_weights('pesos2.h5')


main()
