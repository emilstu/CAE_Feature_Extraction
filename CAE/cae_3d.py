from utils import util, extract_patches
import os
import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json 
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
import random
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
#import pandas as pd



class CAE_3D:
    def __init__(self, model_name, patch_size, min_labeled_pixels, test_size, max_patches, input_dir, out_dir):
        """
        model_name (string): Name of model
        patch_size (tuple): The size of the patches, e.g. (1, 48, 48)
        patch_overlap (tuple): The overlap between patches, e.g. (1, 18, 18)
        test_size (float): The test-train split percentage - e.g. 0.1 for using 10 % of the date for testing and 90 % for training 
        img_dir (string): Direcorty of images
        seg_dir (string): Direcorty of segmentations
        out_dir (string): Where to store the results from training and predicting
        
        """
        self.model_name = model_name
        self.patch_size = patch_size
        self.test_size = test_size
        self.min_labeled_pixels = min_labeled_pixels
        self.max_patches = max_patches
        self.input_dir = input_dir
        self.out_dir = out_dir
        
        self.autoencoder = Model()
        self.patch_train_list = []
        self.patch_test_list = []
        self.patch_train = []
        self.patch_test = []

        self.img_filenames = util.get_paths_from_tree(input_dir, 'imaging')
        self.seg_filenames = util.get_paths_from_tree(input_dir, 'segmentation')


    
    def load():
        #Load saved patches
        #self.autoencoder = util.load_cae_model(self.model_name, self.model_dir)
        pass

    def extract_patches(self):
        # Extract patches and save to disk
        img_patches, _ = extract_patches.extract_3d_patches(    img_filenames=self.img_filenames,
                                                                seg_filenames=self.seg_filenames,
                                                                patch_size=self.patch_size,
                                                                max_patches=self.max_patches    )

        img_patches = np.asarray(img_patches)
        print(img_patches.shape)
        
        # Split into training and testing 
        self.patch_train, self.patch_test = train_test_split(img_patches,test_size=self.test_size)
        
        
        #Save split
        #util.save_sample_list(self.patch_train_list, 'train_samples', self.out_dir)
        #util.save_sample_list(self.patch_test_list, 'test_samples', self.out_dir)


    def train(self, batch_size, epochs, batches_per_epoch):  
        random.shuffle(self.patch_train_list)
        random.shuffle(self.patch_test_list)

        # Calculate number of samples 
        num_train_samples = batches_per_epoch * batch_size
        num_test_samples = int(self.test_size*num_train_samples)
        
        #self.patch_train = util.load_patches(self.patch_train_list[:num_train_samples])
        #self.patch_test = util.load_patches(self.patch_test_list[:num_test_samples])
    
        # Print some useful info 
        print(f'\nNumber of training samples: {num_train_samples}')
        print(f'Number of test samples: {num_test_samples}')
        print(f'train data shape: {self.patch_train.shape}')
        print(f'test data shape: {self.patch_test.shape}')

        print('\nBefore normalizing: ')
        print(f'Pixel max-value train data: {np.max(self.patch_train)}')
        print(f'Pixel max-value test data: {np.max(self.patch_test)}')
        print('\n')

        # Normalize and reshape from (num_samples, 1, y, z) to (num_samples, y, z, 1)
        self.patch_train = util.normalize_data(self.patch_train)
        self.patch_test = util.normalize_data(self.patch_test)
        
        # Print some useful info
        print('\nAfter normalizing: ')
        print(f'Pixel max-value train data: {np.max(self.patch_train)}')
        print(f'Pixel max-value test data: {np.max(self.patch_train)}')
        print(f'\ninput shape = ({self.patch_size[1]}, {self.patch_size[2]}, 1)')

        inp = Input(shape=(self.patch_size[0], self.patch_size[0], self.patch_size[0], 1))
        e = Convolution3D(16, (5, 5, 5), activation='elu', padding='same')(inp)
        e = MaxPooling3D((2, 2, 2), padding='same')(e)
        e = Flatten()(e)
        encoded = Dense(512, activation="elu")(e)
        d = Dense(65536, activation="elu")(encoded)
        d = Reshape((16, 16, 16, 16))(d)
        d = UpSampling3D((2, 2, 2))(d)
        decoded = Convolution3D(1, (5, 5, 5), activation='elu', padding='same')(d)
        print("shape of decoded: ")
        #print(K.int_shape(decoded))

        """
        inp = Input((self.patch_size[1], self.patch_size[2],1))
        e = Conv2D(16, (5, 5), activation='elu', padding='same')(inp)
        e = MaxPooling2D((2, 2))(e)
        #e = Reshape((4608,1,1))(e)
        e = Flatten()(e)
        encoded = Dense(512, activation="elu")(e)
        d = Dense(18432, activation="elu")(encoded)
        d = Reshape((24,24,32))(d)
        d = UpSampling2D((2, 2))(d)
        decoded = Conv2DTranspose(1,(5, 5), strides=1, activation='elu', padding='same')(d)
        """

        self.autoencoder = Model(inp, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='adam', loss='mse')

        history = self.autoencoder.fit( self.patch_train, 
                                        self.patch_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        validation_data=(self.patch_test, self.patch_test)  )


        # Save model
        util.save_cae_model(self.autoencoder, self.model_name, self.out_dir)

        # Save training history 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Fitting Curve during Training')
        plt.ylabel('Loss Function')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(self.out_dir + 'fitting_curve.png')


    def predict(self, batch_size):
        pass


    def delete_patches(self):
        pass






