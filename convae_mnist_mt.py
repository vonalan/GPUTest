import itertools 

import numpy as np 
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.datasets import mnist, cifar10
from keras import backend as keras
from keras.optimizers import Adam
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.metrics import categorical_accuracy


def build_softmax_model():
   inputs = Input(shape=(7*7*32,))
   outputs = Dense(10, activation='softmax')(inputs)
   model = Model(inputs, outputs)
   return model

def build_convae_model():
    inputs = Input(shape=(28,28,1))
    x = inputs

    x = Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x_encode = Flatten()(x)
    encoder = Model(inputs, x_encode)

    x_softmax = Dense(10, activation='softmax')(x_encode)
    softmax_model = Model(inputs, x_softmax)

    x = Reshape((7,7,32))(x_encode)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)

    x_decode = x
    decoder = None
    convae = Model(inputs, x_decode)

    mix_model = Model([inputs], [x_softmax, x_decode])

    return encoder, decoder, convae, softmax_model, mix_model

def run():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    x_train = x_train/255.0
    x_test = x_test/255.0

    temp = np.zeros((y_train.shape[0],10))
    for i in range(y_train.shape[0]):
        temp[i, y_train[i]] = 1
    y_train = temp

    temp = np.zeros((y_test.shape[0],10))
    for i in range(y_test.shape[0]):
        temp[i, y_test[i]] = 1
    y_test = temp

    encoder, _, convae, encoder_softmax_model, mix_model = build_convae_model()
    classifier = build_softmax_model()

    # # convae, softmax
    # convae.compile(optimizer=Adam(), loss=mean_squared_error, metrics=[mean_squared_error])
    # classifier.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    # for i in range(100):
        # convae.fit(x_train, x_train, epochs=1, verbose=0)
        # x_decode_train = encoder.predict(x_train)
        # x_decode_test = encoder.predict(x_test)
        # classifier.fit(x_decode_train, y_train, epochs=1, verbose=0)
        # print(classifier.evaluate(x_decode_test, y_test, verbose=0))
    
    # encoder_softmax
    encoder_softmax_model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    for i in range(100):
        encoder_softmax_model.fit(x_train, y_train, epochs=1, verbose=0)
        print(encoder_softmax_model.evaluate(x_test, y_test, verbose=0))

    '''
    # mix_model
    mix_model.compile(optimizer=Adam(), loss=[categorical_crossentropy, mean_squared_error], metrics=[categorical_accuracy, mean_squared_error])
    for i in range(100):
        mix_model.fit([x_train], [y_train, x_train], epochs=1, verbose=0)
        print(mix_model.evaluate([x_test], [y_test, x_test], verbose=0))
    '''
    
if __name__ == '__main__':
    with tf.device('/cpu:0'):
        run()


        
        



