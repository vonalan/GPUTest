import itertools 

import numpy as np 
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model 
from keras.datasets import mnist 
from keras.optimizers import Adam
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.metrics import categorical_accuracy

def build_softmax_model():
   inputs = Input(shape=(200,))
   outputs = Dense(10, activation='softmax')(inputs)
   model = Model(inputs, outputs)
   return model

def build_ae_model():
    inputs = Input(shape=(32*32*3,))
    
    x_encoder = Dense(1024, activation='relu', use_bias=True)(inputs)
    x_encoder = Dense(200, activation='relu', use_bias=True)(x_encoder)
    x_decoder = Dense(1024, activation='relu', use_bias=True)(x_encoder)
    x_decoder = Dense(32*32*3, activation='relu', use_bias=True)(x_decoder)
    outputs = x_decoder 
    
    encoder = Model([inputs], [x_encoder])
    decoder = None
    ae_model = Model([inputs], [outputs])
    
    x_softmax = Dense(10, activation='softmax')(x_encoder)
    softmax_model = Model([inputs], [x_softmax])
    return encoder, decoder, ae_model, softmax_model

def main(): 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 32*32*3))
    x_test = np.reshape(x_test, (-1, 32*32*3))
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
    
    encoder, _, ae, _ = build_convae_model()
    ae.compile(optimizer=Adam(), loss=[mean_squared_error], metrics=[mean_squared_error])

    classifier = build_softmax_model()
    classifier.compile(optimizer=Adam(), loss=[categorical_crossentropy], metrics=[categorical_accuracy])
    
    for i in range(100):
        ae.fit(x_train, x_train, epochs=1, verbose=0)
        x_decode_train = encoder.predict(x_train)
        x_decode_test = encoder.predict(x_test)
        classifier.fit(x_decode_train, y_train, epochs=1, verbose=0)
        print(classifier.evaluate(x_decode_test, y_test, verbose=0))

if __name__ == '__main__': 
    main()



