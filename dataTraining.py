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
    train_path = 'food-11/training'
    valid_path = 'food-11/validation'

    #train_path = Path("C:/Users/victo/OneDrive/Documentos/TCC/Food-11/training")
    #valid_path = Path("C:/Users/victo/OneDrive/Documentos/TCC/Food-11/validation")
    train_batches = ImageDataGenerator(rescale=1./223).flow_from_directory(train_path, target_size=(224,224),  
    classes=['bread','dairy','dessert' , 'egg'  ,'fried',  'meat',  'pasta',  'rice',  'seafood',  'soup',  'vegetables'
], batch_size=50 )

    valid_batches = ImageDataGenerator(rescale=1./223).flow_from_directory(valid_path, target_size=(224,224),  
        classes=['bread','dairy','dessert' , 'egg'  ,'fried',  'meat',  'pasta',  'rice',  'seafood',  'soup',  'vegetables'
    ], batch_size=50 )

    # model = Sequential([
    #     Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)),
    #     Flatten(),
    #     Dense(11,activation='softmax'),
    # ])

    vgg16_model = keras.applications.vgg16.VGG16()
    # vgg16_model = keras.application.vgg16.VGG16()
    # vgg16_model.summary()

    type(vgg16_model)

    # 

    model = Sequential()
    i=0;
    for layer in vgg16_model.layers:
        i+=1
        if(i<=22):
            model.add(layer)

    model.summary()
    model.layers.pop()
    # model.layers.pop()
    # model.layers.pop()
    # model.add(Dense(11, activation="softmax"))

    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []

    for layer in model.layers:
        layer.trainable = False

    model.summary()

    model.add(Dense(11, activation="softmax"))
    model.summary()


    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_batches, steps_per_epoch=200, validation_data=valid_batches, validation_steps=100, epochs=50,verbose=2)

    model.save_weights('pesos.h5')


main()
