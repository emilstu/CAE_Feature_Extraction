from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam

class CAE_2D:
    def create_model(self, patch_size):
        inp = Input((patch_size[1], patch_size[2],1))
        e = Conv2D(32, (5, 5), activation='elu', padding='same')(inp)
        e = BatchNormalization()(e)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(32, (5, 5), activation='elu', padding='same')(e)
        e = BatchNormalization()(e)
        e = MaxPooling2D((2, 2))(e)
        e = Flatten()(e)
        #e = Dropout(0.4)(e)
        encoded = Dense(512, activation="elu")(e)
        #d = Dropout(0.4)(encoded)
        d = Dense(int(patch_size[1]/4)*int(patch_size[2]/4)*32, activation="elu")(encoded)
        #d = Dropout(0.4)(d)
        d = Reshape((int(patch_size[1]/4),int(patch_size[2]/4),32))(d)
        d = UpSampling2D((2, 2))(d)
        d = Conv2DTranspose(32,(5, 5), strides=1, activation='elu', padding='same')(d)
        d = BatchNormalization()(d)
        d = UpSampling2D((2, 2))(d)
        decoded = Conv2DTranspose(1,(5, 5), strides=1, activation='sigmoid', padding='same')(d)

        autoencoder = Model(inp, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer='adam', loss='mse')
        #autoencoder.compile(optimizer=SGD(lr=0.1, nesterov=True), loss='mse')        
        
        return autoencoder
       
    def create_model_2(self, patch_size):
        inp = Input((patch_size[1], patch_size[2],1))
        e = Conv2D(32, (5, 5), activation='elu', padding='same')(inp)
        e = BatchNormalization()(e)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(32, (5, 5), activation='elu', padding='same')(inp)
        e = BatchNormalization()(e)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(32, (5, 5), activation='elu', padding='same')(inp)
        e = BatchNormalization()(e)
        e = MaxPooling2D((2, 2))(e)
        e = Flatten()(e)
        encoded = Dense(512, activation="elu")(e)
        #d = Dropout(0.4)(encoded)
        d = Dense(int(patch_size[1]/2)*int(patch_size[2]/2)*32, activation="elu")(encoded)
        #d = Dropout(0.4)(d)
        d = Reshape((int(patch_size[1]/2),int(patch_size[2]/2),32))(d)
        d = UpSampling2D((2, 2))(d)
        d = Conv2D(1,(5, 5), strides=1, activation='elu', padding='same')(d)
        d = UpSampling2D((2, 2))(d)
        decoded = Conv2D(1,(5, 5), strides=1, activation='sigmoid', padding='same')(d)
        #sigmoid

        autoencoder = Model(inp, decoded, name="2D_CAE")
        autoencoder.summary()
        autoencoder.compile(loss='mean_squared_error', optimizer='adam', lr=0.001)
        #optimizer = RMSprop()
        #autoencoder.compile(optimizer='adam', loss='mse')
        #autoencoder.compile(optimizer=SGD(lr=0.1, nesterov=True), loss='mean_squared_error')        
        
        return autoencoder
