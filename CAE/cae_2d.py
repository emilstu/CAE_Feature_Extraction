from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adamax
from tensorflow.keras import regularizers
import numpy as np

class CAE_2D:
    def create_model(self, patch_size, pixel_norm):
        if pixel_norm=='z-score': act='linear'
        if pixel_norm=='minmax': act='sigmoid'
        if pixel_norm=='abs': act='tanh'

        ps=np.array(patch_size, dtype=np.int16)
        ps=(ps[ps != 1])
        input_shape=(ps[0], ps[1], 1)

        inp = Input(input_shape)
        e = Conv2D(16, (5, 5), strides=2, activation='elu', padding='same')(inp)
        e = BatchNormalization()(e)
        #e = MaxPooling2D((2, 2))(e)
        #e = Conv2D(16, (5, 5), strides=2, activation='elu', padding='same')(e)
        #e = BatchNormalization()(e)
        #e = MaxPooling2D((2, 2))(e)
        e = Flatten()(e)
        #encoded = Dense(512, activation="elu", activity_regularizer=regularizers.l1(10e-5), kernel_initializer='he_uniform')(e)
        encoded = Dense(512, activation="elu")(e)
        d = Dropout(0.4)(encoded)
        d = Dense(int(input_shape[0]/2)*int(input_shape[1]/2)*16, activation="elu")(d)
        #d = Dropout(0.1)(d)
        d = Reshape((int(input_shape[0]/2),int(input_shape[1]/2),16))(d)
        #d = UpSampling2D((2, 2))(d)
        #d = Conv2DTranspose(32,(5, 5), strides=2, activation='elu', padding='same')(d)
        #d = BatchNormalization()(d)
        #d = UpSampling2D((2, 2))(d)
        decoded = Conv2DTranspose(1,(5, 5), strides=2, activation=act, padding='same')(d)
        autoencoder = Model(inp, decoded)
        autoencoder.summary()
        optimizer=Adam(learning_rate=0.001)
        #optimizer=Adamax(learning_rate=0.01)
        #autoencoder.compile(optimizer=optimizer, loss='mse')
        autoencoder.compile(optimizer=optimizer, loss='mse')        
        
        return autoencoder