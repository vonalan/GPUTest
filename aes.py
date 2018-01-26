import itertools 

import numpy as np 
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model 

def build_softmax_model():
   inputs = Input(shape=(200,))
   outputs = Dense(10, activation='softmax')(inputs)
   model = Model(inputs, outputs)
   return model

def build_ae_model(activation=''):
    inputs = Input(shape=(784,))
    
    x_encoder = Dense(512, activation=activation, use_bias=True)(inputs)
    x_encoder = Dense(200, activation=activation, use_bias=True)(x_encoder)
    x_decoder = Dense(512, activation=activation, use_bias=True)(x_encoder)
    x_decoder = Dense(784, activation=activation, use_bias=True)(x_decoder)
    outputs = x_decoder 
    
    encoder = Model(inputs, x_encoder)
    decoder = None
    ae_model = Model(inputs, outputs)
    
    x_softmax = Dense(10, activation='softmax')(x_encoder)
    softmax_model = Model(inputs, x_softmax)
    return encoder, decoder, ae_model, softmax_model

def run_ae(activation=''): 
    encoder, _, ae, _ = build_ae_model(activation=activation)
    ae.compile(optimizer=xxx, loss=xxx, metrics=[xxx])
    ae.train(xtrain, xtrain)
    xtrain = encoder.predict(xtrain)
    xtest = encoder.predict(xtest)
    return xtrain, xtest

def run_embeded_aes(activations=None):
    XList = [run_ae(activation) for activation in activations]
    encode_xtrain = [item[0] for item in XList]
    encode_xtesta = [item[1] for item in XList]

    encode_xtrain = np.array(encode_xtrain).mean(axis=0)
    encode_xtesta = np.array(encode_xtesta).mean(axis=0)
    
    cls = build_softmax_model()
    cls.compile(optimizer=xxx, loss=xxx, metrics=[xxx])
    loss_train, acc_train = cls.train(encode_xtrain, ytrain)
    loss_test, acc_test = cls.evaluate(encode_xtest, ytest)
    return loss_test, acc_test

def main(): 
    _, acc_sigmoid_aes = run_embeded_aes(['sigmoid'] * 4)
    _, acc_tahn_aes = run_embeded_aes(['tanh'] * 4)
    _, acc_relu_aes = run_embeded_aes(['relu'] * 4)
    _, acc_xxx_aes = run_embeded_aes(['softplus'] * 4)
    _, acc_mix_aes  = run_embeded_aes(['sigmoid', 'tanh', 'relu', 'softplus'])

if __name__ == '__main__': 
    main()


        
        



