from utils import util, extract_patches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json 
from keras.layers import Dense, Dropout, Input, Reshape, Flatten 
import random
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D, Conv3DTranspose
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import regularizers
from keras.optimizers import Adam, SGD, Adamax


class CAE_3D:
    def create_model(self, patch_size, pixel_norm):  
        if pixel_norm=='z-score': act='linear'
        if pixel_norm=='minmax': act='sigmoid'
        if pixel_norm=='abs': act='tanh'

        inp = Input(shape=(patch_size[0], patch_size[0], patch_size[0], 1))
        e = Convolution3D(16, (5, 5, 5), activation='elu', strides=1, padding='same')(inp)
        e = BatchNormalization()(e)
        e = MaxPooling3D((2, 2, 2), padding='same')(e)
        e = Convolution3D(32, (5, 5, 5), activation='elu', strides=2, padding='same')(e)
        e = BatchNormalization()(e)
        #e = MaxPooling3D((2, 2, 2), padding='same')(e)
        #e = Convolution3D(16, (5, 5, 5), activation='elu', padding='same')(e)
        #e = BatchNormalization()(e)
        #e = MaxPooling3D((2, 2, 2), padding='same')(e)
        e = Flatten()(e)
        #encoded = Dense(512, activation="elu", activity_regularizer=regularizers.l1(10e-5), kernel_initializer='he_uniform')(e)
        encoded = Dense(512, activation="elu")(e)
        #d = Dropout(0.2)(encoded)
        d = Dense(int(patch_size[0]/4)*int(patch_size[1]/4)*int(patch_size[2]/4)*32, activation="elu")(encoded)
        #d = Dropout(0.4)(d)
        d = Reshape((int(patch_size[0]/4), int(patch_size[1]/4), int(patch_size[2]/4), 32))(d)
        #d = UpSampling3D((2, 2, 2))(d)
        d = Conv3DTranspose(16, (5, 5, 5), strides=2, activation='elu', padding='same')(d)
        d = BatchNormalization()(d)
        d = UpSampling3D((2, 2, 2))(d)
        #d = Convolution3D(64, (5, 5, 5), activation='elu', padding='same')(d)
        #d = BatchNormalization()(d)
        #d = UpSampling3D((2, 2, 2))(d)
        decoded = Conv3DTranspose(1, (5, 5, 5), strides=1, activation=act, padding='same')(d)

        autoencoder = Model(inp, decoded)
        autoencoder.summary()
        
        #optimizer=SGD(lr=0.1, nesterov=True)
        #optimizer=Adamax(learning_rate=0.001)
        optimizer=Adam(learning_rate=0.01)
        autoencoder.compile(optimizer=optimizer, loss='mse')   
        
        return autoencoder
