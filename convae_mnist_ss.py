import itertools 

import numpy as np 
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
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
    raw_rgb = Input(shape=(28,28,1))
    mask = Input(shape=(28,28,1))

    x = Concatenate(axis=3)([raw_rgb, mask])
    # x = inputs

    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x_encode = Flatten()(x)
    encoder = Model([raw_rgb, mask], [x_encode])

    x_clsf = Dense(10+1, activation='softmax')(x_encode)
    clsf_model = Model([raw_rgb, mask], [x_clsf])

    # x_clst = Dense(10, activation='softmax')(x_encode)
    # x_clst_loss = cluster_loss_layer()(x_clst)
    # clsf_model = Model([raw_rgb, mask], [x_clst_loss])

    x = Reshape((7,7,64))(x_encode)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)

    x_decode = x
    decoder = None
    convae = Model([raw_rgb, mask], [x_decode])

    mix_model = Model([raw_rgb, mask], [x_decode, x_clsf])

    return encoder, decoder, convae, clsf_model, mix_model

def run():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    x_train = x_train/255.0
    x_test = x_test/255.0

    temp = np.zeros((y_train.shape[0],10+1))
    for i in range(y_train.shape[0]):
        temp[i, y_train[i]] = 1
    y_train = temp

    temp = np.zeros((y_test.shape[0],10+1))
    for i in range(y_test.shape[0]):
        temp[i, y_test[i]] = 1
    y_test = temp

    m_train = np.zeros(x_train.shape) + 1
    m_test = np.zeros(x_test.shape) + 1

    m = x_train.shape[0]
    idx = np.arange(m)
    np.random.shuffle(idx)

    m_train[idx[int(m * 0.1):],...] = -1
    y_train[idx[int(m * 0.1):],...] = 0
    y_train[idx[int(m * 0.1):],-1] = 1

    # print(x_train[idx[int(m * 0.5)-1],...].flatten())
    # print(m_train[idx[int(m * 0.5)-1],...].flatten())
    # print(y_train[idx[int(m * 0.5)-1],...].flatten())

    # print(x_train[idx[int(m * 0.5)],...].flatten())
    # print(m_train[idx[int(m * 0.5)],...].flatten())
    # print(y_train[idx[int(m * 0.5)],...].flatten())

    encoder, _, convae, encoder_softmax_model, mix_model = build_convae_model()
    classifier = build_softmax_model()

    # # convae, softmax
    # convae.compile(optimizer=Adam(), loss=mean_squared_error, metrics=[mean_squared_error])
    # classifier.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    # for i in range(10):
        # convae.fit(x_train, x_train, epochs=1, verbose=0)
        # x_decode_train = encoder.predict(x_train)
        # x_decode_test = encoder.predict(x_test)
        # classifier.fit(x_decode_train, y_train, epochs=1, verbose=0)
        # print(classifier.evaluate(x_decode_test, y_test, verbose=0))


    # # encoder_softmax
    # encoder_softmax_model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])
    # for i in range(10):
    #     encoder_softmax_model.fit(x_train, y_train, epochs=1, verbose=0)
    #     print(encoder_softma x_model.evaluate(x_test, y_test, verbose=0))

    # mix_model
    mix_model.compile(optimizer=Adam(), loss=[mean_squared_error, categorical_crossentropy], metrics=[mean_squared_error, categorical_accuracy])
    for i in range(100):
        mix_model.fit([x_train, m_train], [x_train, y_train], epochs=1, verbose=0)
        print(mix_model.evaluate([x_test, m_test], [x_test, y_test], verbose=0))

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        run()


        
        



