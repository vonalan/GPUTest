import itertools 

import numpy as np 
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model 
from keras.datasets import mnist 
from keras.optimizers import Adam
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.metrics import categorical_accuracy

# global variables 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (-1, 28*28*1))
x_test = np.reshape(x_test, (-1, 28*28*1))
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
    

def build_softmax_model():
   inputs = Input(shape=(200,))
   outputs = Dense(10, activation='softmax')(inputs)
   model = Model(inputs, outputs)
   return model

def build_ae_model(activation=''):
    inputs = Input(shape=(784,))
    
    x_encoder = Dense(200, activation=activation, use_bias=True)(inputs)
    x_decoder = Dense(784, activation=activation, use_bias=True)(x_encoder)
    outputs = x_decoder 
    
    encoder = Model([inputs], [x_encoder])
    decoder = None
    ae_model = Model([inputs], [outputs])
    
    x_softmax = Dense(10, activation='softmax')(x_encoder)
    softmax_model = Model([inputs], [x_softmax])
    return encoder, decoder, ae_model, softmax_model

def run_ae(activation=''): 
    encoder, _, ae, _ = build_ae_model(activation=activation)
    ae.compile(optimizer=Adam(), loss=[mean_squared_error], metrics=[mean_squared_error])
    
    # bugs | new name 
    xtrain = x_train 
    xtest = x_test 
    if activation == 'tanh': 
        xtrain = x_train * 2 - 1 
        xtest = x_test * 2 - 1
    
    ae.fit(xtrain, xtrain, epochs=1, verbose=0)
    xtrain = encoder.predict(xtrain)
    xtest = encoder.predict(xtest)
    return xtrain, xtest

def run_aes(activations=None):
    XList = [run_ae(activation) for activation in activations]
    encode_xtrain = [item[0] for item in XList]
    encode_xtesta = [item[1] for item in XList]
    
    '''
    # bugs | average 
    encode_xtrain = np.array(encode_xtrain).mean(axis=0)
    encode_xtesta = np.array(encode_xtesta).mean(axis=0)
    '''
    
    # bugs | concatenate 
    print(np.array(encode_xtrain).shape, np.array(encode_xtrain).shape)
    encode_xtrain = np.array(encode_xtrain).transpose((1,0,2)).reshape((-1, 200 * 4))
    encode_xtesta = np.array(encode_xtesta).transpose((1,0,2)).reshape((-1, 200 * 4))
    print(encode_xtrain.shape, encode_xtest.shape)
    
    cls = build_softmax_model()
    cls.compile(optimizer=Adam(), loss=[categorical_crossentropy], metrics=[categorical_accuracy])
    loss_train, acc_train = cls.fit(encode_xtrain, ytrain, epochs=1, verbose=0)
    loss_test, acc_test = cls.evaluate(encode_xtest, ytest, verbose=0)
    return loss_test, acc_test

def main():
    _, acc_sigmoid_aes = run_aes(['sigmoid'] * 4)
    _, acc_tahn_aes = run_aes(['tanh'] * 4)
    _, acc_relu_aes = run_aes(['relu'] * 4)
    _, acc_softplus_aes = run_aes(['softplus'] * 4)
    _, acc_mix_aes  = run_aes(['sigmoid', 'tanh', 'relu', 'softplus'])
    print(acc_sigmoid_aes, acc_tahn_aes, acc_relu_aes, acc_softplus_aes, acc_mix_aes)
    
if __name__ == '__main__': 
    main()


        
        



