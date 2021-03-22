from utils import util, extract_patches
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json 
from keras.layers import Dense, Dropout, Input, Reshape, Flatten 
import random
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import tensorflow as tf


class CAE_3D:
    def create_model(self, patch_size):  
        inp = Input(shape=(patch_size[0], patch_size[0], patch_size[0], 1))
        e = Convolution3D(16, (5, 5, 5), activation='elu', padding='same')(inp)
        e = BatchNormalization()(e)
        e = MaxPooling3D((2, 2, 2), padding='same')(e)
        e = Flatten()(e)
        e = Dropout(0.5)(e)
        encoded = Dense(512, activation="elu")(e)
        d = Dropout(0.5)(encoded)
        d = Dense(int(patch_size[0]/2)*int(patch_size[1]/2)*int(patch_size[2]/2)*16, activation="elu")(d)
        d = Dropout(0.5)(d)
        d = Reshape((int(patch_size[0]/2), int(patch_size[1]/2), int(patch_size[2]/2), 16))(d)
        d = UpSampling3D((2, 2, 2))(d)
        decoded = Convolution3D(1, (5, 5, 5), activation='sigmoid', padding='same')(d)
    
        autoencoder = Model(inp, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder


