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
import shutil
#import pandas as pd



class CAE_2D:
    def __init__(self, model_name, patch_size, patch_overlap, min_labeled_pixels, test_size, input_dir, out_dir):
        """
        model_name (string): Name of model
        patch_size (tuple): The size of the patches, e.g. (1, 48, 48)
        patch_overlap (tuple): The overlap between patches, e.g. (1, 18, 18)
        min_labeled_pixels (float): The minimum percentage of labeled voxels in a patch (float between 0 and 1)
        test_size (float): The test-train split percentage - e.g. 0.1 for using 10 % of the date for testing and 90 % for training 
        img_dir (string): Direcorty of images
        seg_dir (string): Direcorty of segmentations
        out_dir (string): Where to store the results from training and predicting
        
        """
        
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_labeled_pixels = min_labeled_pixels
        self.test_size = test_size
        self.input_dir = input_dir
        self.out_dir = out_dir
        
        self.autoencoder = Model()
        self.patch_train_list = []
        self.patch_test_list = []
        self.patch_train = []
        self.patch_test = []

        self.img_filenames = util.get_paths_from_tree(input_dir, 'imaging')
        self.seg_filenames = util.get_paths_from_tree(input_dir, 'segmentation')

    
    def load(self, model_dir):
        self.load_patch_list_from_disk(model_dir)
        self.load_model_from_disk(self.model_name, model_dir)
        



    def extract_patches(self):
        # Extract patches and save to disk
        img_patches, _, _ = extract_patches.extract_2d_patches( img_filenames=self.img_filenames,
                                                                seg_filenames=self.seg_filenames,
                                                                patch_size=self.patch_size,
                                                                patch_overlap=self.patch_overlap,
                                                                min_labeled_pixels=self.min_labeled_pixels  )
        
         
        self.patch_train_list, self.patch_test_list = train_test_split(img_patches,test_size=self.test_size)
        
        #Save split
        util.save_sample_list(self.patch_train_list, 'train_samples', self.out_dir)
        util.save_sample_list(self.patch_test_list, 'test_samples', self.out_dir)


    def train(self, batch_size, epochs, batches_per_epoch):  
        # Select samples for training and testing (100 000 for training and 10 000 for testing --> 200 and 20 minibatches pr epoch)
        random.shuffle(self.patch_train_list)
        random.shuffle(self.patch_test_list)

        num_train_samples = batches_per_epoch * batch_size
        num_test_samples = int(self.test_size*num_train_samples)
        
        self.patch_train = util.load_patches(self.patch_train_list[:num_train_samples])
        self.patch_test = util.load_patches(self.patch_test_list[:num_test_samples])
        
        # Print some useful info 
        print(f'\nNumber of training samples: {num_train_samples}')
        print(f'Number of test samples: {num_test_samples}')
        print(f'train data shape: {self.patch_train.shape}')
        print(f'test data shape: {self.patch_test.shape}')

        print('\nBefore normalizing: ')
        print(f'Pixel max-value train data: {np.max(self.patch_train)}')
        print(f'Pixel max-value test data: {np.max(self.patch_train)}\n')

        # Normalize data
        self.patch_train = util.normalize_data(self.patch_train)
        self.patch_test = util.normalize_data(self.patch_test)

        # Reshape data from (num_samples, 1, 48, 48) to (num_samples, 48, 48, 1)
        self.patch_train = util.reshape_data_2d(self.patch_train, self.patch_size)
        self.patch_test = util.reshape_data_2d(self.patch_test, self.patch_size)
        
        # Print some useful info
        print('After normalizing: ')
        print(f'Pixel max-value train data: {np.max(self.patch_train)}')
        print(f'Pixel max-value test data: {np.max(self.patch_train)}')
        print(f'\ninput shape = ({self.patch_size[1]}, {self.patch_size[2]}, 1)')

        inp = Input((self.patch_size[1], self.patch_size[2],1))
        e = Conv2D(16, (5, 5), activation='elu', padding='same')(inp)
        e = MaxPooling2D((2, 2))(e)
        #e = Reshape((4608,1,1))(e)
        e = Flatten()(e)
        encoded = Dense(512, activation="elu")(e)
        d = Dense(int(self.patch_size[1]/2)*int(self.patch_size[2]/2)*32, activation="elu")(encoded)
        d = Reshape((int(self.patch_size[1]/2),int(self.patch_size[2]/2),32))(d)
        d = UpSampling2D((2, 2))(d)
        decoded = Conv2DTranspose(1,(5, 5), strides=1, activation='elu', padding='same')(d)

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
        # Make predictions 
        pred = self.autoencoder.predict(self.patch_test, verbose=1, batch_size=batch_size)
        print(pred.shape)
        
        # Save results to csv
        #mse = np.mean(np.power(patch_test - pred, 2), axis=1)
        #error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': patch_test}, index=[0])
        #error_df.to_csv(out_dir + "results.csv", sep='\t')

        # Save results to image 
        plt.figure(figsize=(20, 4))
        print("Test Images")
        for i in range(10):
            plt.subplot(2, 10, i+1)
            plt.imshow(self.patch_test[i, ..., 0], cmap='gray')
        #plt.show()
        plt.savefig(self.out_dir + 'org.png')    

        plt.figure(figsize=(20, 4))
        print("Reconstruction of Test Images")
        for i in range(10):
            plt.subplot(2, 10, i+1)
            plt.imshow(pred[i, ..., 0], cmap='gray')  
        #plt.show()
        plt.savefig(self.out_dir + 'rec.png')


    def load_model_from_disk(self, model_dir): 
        # Load model and weights 
        self.autoencoder = util.load_cae_model(self.model_name, model_dir)

        # Get patches for prediction
        self.patch_test = util.load_patches(self.patch_test_list)

        # Normalize data
        self.patch_test = util.normalize_data(self.patch_test)

        # Reshape data from (num_samples, 1, 48, 48) to (num_samples, 48, 48, 1)
        self.patch_test = util.reshape_data_2d(self.patch_test, self.patch_size)

    def load_patch_list_from_disk(self, model_dir):
        self.patch_train_list = util.load_sample_list('train_samples', model_dir)
        self.patch_test_list= util.load_sample_list('test_samples', model_dir)


    def delete_patches(self):
        print('\nDeleting patches from disk..')
        shutil.rmtree('tmp/CAE/2D/patches/img/')
        shutil.rmtree('tmp/CAE/2D/patches/seg/')





