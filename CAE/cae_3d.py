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
from keras.callbacks import EarlyStopping
import tensorflow as tf



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
                                                                labelmap_filenames=self.seg_filenames,
                                                                patch_size=self.patch_size,
                                                                max_patches=self.max_patches,    
                                                                extract_labelmap_patches=False  )
        print(img_patches.shape)
        
        # Split into training and testing 
        self.patch_train, self.patch_test = train_test_split(img_patches,test_size=self.test_size, random_state=55)
        


    def train(self, batch_size, epochs):  
        inp = Input(shape=(self.patch_size[0], self.patch_size[0], self.patch_size[0], 1))
        e = Convolution3D(16, (5, 5, 5), activation='elu', padding='same')(inp)
        e = MaxPooling3D((2, 2, 2), padding='same')(e)
        e = Flatten()(e)
        e = Dropout(0.5)(e)
        encoded = Dense(512, activation="elu")(e)
        d = Dropout(0.5)(encoded)
        d = Dense(int(self.patch_size[0]/2)*int(self.patch_size[1]/2)*int(self.patch_size[2]/2)*16, activation="elu")(d)
        d = Dropout(0.5)(d)
        d = Reshape((int(self.patch_size[0]/2), int(self.patch_size[1]/2), int(self.patch_size[2]/2), 16))(d)
        d = UpSampling3D((2, 2, 2))(d)
        decoded = Convolution3D(1, (5, 5, 5), activation='elu', padding='same')(d)
    
        self.autoencoder = Model(inp, decoded)
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='adam', loss='mse')
        es_callback = EarlyStopping(monitor='val_loss', patience=50)

        history = self.autoencoder.fit( self.patch_train, 
                                        self.patch_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        validation_data=(self.patch_test, self.patch_test),
                                        callbacks=[es_callback]  )


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
        
         # Calculate reconstruction error
        recon_error = tf.square(pred - self.patch_test)

        # Convert to numpy and normalize
        recon_error = recon_error.numpy()

        # Calculate and save mse
        mse = np.mean(recon_error)

        # Normalize for visualization
        recon_error *= 255.0/recon_error.max()

        # Get random slice number
        slice_num = np.random.randint(pred.shape[1]-1)

        # Save results to image 
        plt.figure(figsize=(30, 4))
        for i in range(10):
            plt.subplot(3, 10, i+1) 
            plt.axis('off')
            plt.imshow(self.patch_test[i, slice_num, ...,0], cmap='gray')
        plt.savefig(self.out_dir + 'org.png')    

        plt.figure(figsize=(30, 4))
        for i in range(10):
            plt.subplot(3, 10, i+1)
            plt.axis('off')
            plt.imshow(pred[i, slice_num, ..., 0], cmap='gray')
        plt.savefig(self.out_dir + 'rec.png')

        plt.figure(figsize=(30, 4))
        for i in range(10):
            plt.subplot(3, 10, i+1)
            plt.axis('off')
            plt.imshow(recon_error[i, slice_num, ..., 0], cmap='hsv')
            plt.clim(0,100)
            plt.colorbar()
        plt.savefig(self.out_dir + 'error.png')

        # Add parameters to dict and save it  
        info = {'patch_size': self.patch_size,
                'max_patches': self.max_patches,
                'batch_size': batch_size,
                'test_size': self.test_size
                }
        
        util.save_dict(info, self.out_dir, 'info.csv')
        util.save_dict({'mse': mse}, self.out_dir, 'evaluation.csv')


    
    def preprocess_data(self, batches_per_epoch, batch_size):
        # Shuffle data
        random.shuffle(self.patch_train_list)
        random.shuffle(self.patch_test_list)

        # Calculate number of samples 
        num_train_samples = batches_per_epoch * batch_size
        num_test_samples = int(self.test_size*num_train_samples)
    
        # Print some useful info 
        print(f'\nNumber of training samples: {num_train_samples}')
        print(f'Number of test samples: {num_test_samples}')
        print(f'train data shape: {self.patch_train.shape}')
        print(f'test data shape: {self.patch_test.shape}')

        print('\nBefore normalizing: ')
        print(f'Pixel max-value train data: {np.max(self.patch_train)}')
        print(f'Pixel max-value test data: {np.max(self.patch_test)}')

        # Normalize and reshape from (num_samples, 1, y, z) to (num_samples, y, z, 1)
        self.patch_train = util.normalize_data(self.patch_train)
        self.patch_test = util.normalize_data(self.patch_test)
        
        # Print some useful info
        print('\nAfter normalizing: ')
        print(f'Pixel max-value train data: {np.max(self.patch_train)}')
        print(f'Pixel max-value test data: {np.max(self.patch_train)}')
        print(f'\ninput shape = ({self.patch_size[0]}, {self.patch_size[1]}, {self.patch_size[2]}, 1)')




