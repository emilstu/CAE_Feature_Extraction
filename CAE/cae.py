from os import path
from CAE.cae_2d import CAE_2D
from CAE.cae_3d import CAE_3D
from utils import util, extract_patches
from utils.data_generator import DataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from matplotlib.colors import LogNorm
import random

class CAE:
    def __init__(self, input_dir, patch_size, batch_size, test_size, prepare_batches):
        self.input_dir = input_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.prepare_batches = prepare_batches
        self.partition = dict()

        self.img_filenames = util.get_paths_from_tree(input_dir, 'imaging')
        self.seg_filenames = util.get_paths_from_tree(input_dir, 'segmentation')
        
        if 1 in patch_size:
            # Create out dirs if it doesn't exists
            if not os.path.exists('evaluation/CAE/2D/'):
                os.makedirs('evaluation/CAE/2D/')
            self.model_name = 'model_2D'
            self.out_dir = util.get_next_folder_name('evaluation/CAE/2D/', pattern='ex')
            os.makedirs(self.out_dir)
            self.cae = CAE_2D()
        else:
            # Create out dirs if it doesn't exists
            if not os.path.exists('evaluation/CAE/3D/'):
                os.makedirs('evaluation/CAE/3D/')
            self.model_name = 'model_3D'
            self.out_dir = util.get_next_folder_name('evaluation/CAE/3D/', pattern='ex')
            os.makedirs(self.out_dir)
            self.cae = CAE_3D()


    def prepare_data(self, sampler_type, max_patches, resample, patch_overlap, min_labeled_voxels, label_prob, clipping, pixel_norm, load_data):
        self.sampler_type = sampler_type
        self.resample = resample
        self.max_patches = max_patches
        self.label_prob = label_prob
        self.clipping=clipping
        self.pixel_norm=pixel_norm
    
        if sampler_type == 'grid':
            n_def = 'not defined for GridSampler'
            self.label_prob = n_def
            self.min_labeled_voxels = min_labeled_voxels
            self.patch_overlap = patch_overlap
        elif sampler_type == 'label':
            n_def = 'not defined for LabelSampler'
            self.min_labeled_voxels = n_def
            self.patch_overlap = n_def
            self.label_prob = label_prob      
        else: 
            raise Exception(f'\nError: sampler_type mist either be "grid" or "label"\n')

        if not load_data:
            #Clip pixels intensities and normalize data
            if self.clipping: self.img_filenames = util.clipping(self.img_filenames, clipping=self.clipping)
            if not self.clipping: self.img_filenames = util.normalize_data(self.img_filenames, norm=self.pixel_norm) 
            
            # Create out dir for patches
            save_dir=util.create_cae_patch_dir(self.prepare_batches, self.patch_size)

            # Extract patches and save to disk
            ids, _ = extract_patches.patch_sampler( img_filenames=self.img_filenames,
                                                    labelmap_filenames=self.seg_filenames,
                                                    patch_size=self.patch_size,
                                                    sampler_type=sampler_type,
                                                    voxel_spacing=self.resample,
                                                    out_dir=save_dir,
                                                    max_patches=self.max_patches,
                                                    patch_overlap=self.patch_overlap,
                                                    pixel_norm=self.pixel_norm,
                                                    min_labeled_voxels=self.min_labeled_voxels,
                                                    label_prob=self.label_prob,
                                                    save_patches=True,
                                                    prepare_batches=self.prepare_batches,  
                                                    batch_size=self.batch_size  )        
            # Split data to training and testing sets 
            train_ids, val_ids = train_test_split(ids,test_size=self.test_size, random_state=19)

            # Add split to partition dict
            self.partition['train'] = train_ids
            self.partition['validation'] = val_ids
            util.save_partition(self.partition, save_dir)
            self.create_generators(self.prepare_batches, save_dir) 
        else:
            # Load patches from disk
            save_dir = util.get_cae_patch_dir(self.prepare_batches, self.patch_size)
            self.partition = util.load_partition(self.prepare_batches, self.patch_size)
            self.create_generators(self.prepare_batches, save_dir)

    def train(self, epochs):
        self.epochs = epochs
        self.autoencoder = self.cae.create_model(self.patch_size, self.pixel_norm)


        es_callback = EarlyStopping(monitor='val_loss', patience=50, mode='min')
        mcp_callback = ModelCheckpoint(f'{self.out_dir}model_best.hdf5', save_best_only=True, monitor='val_loss')
        rd_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, min_lr=1e-6)

        multi=False
        num_workers=1
        #if not self.prepare_batches: 
        #    multi = True
        #    num_workers=2
              
        history = self.autoencoder.fit( self.training_generator,
                                        epochs=self.epochs,
                                        validation_data=self.validation_generator,
                                        callbacks=[es_callback, mcp_callback, rd_callback],
                                        workers=num_workers, use_multiprocessing=multi )

        # Save model and training history 
        util.save_cae_model(self.autoencoder, self.model_name, self.out_dir)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Fitting Curve during Training')
        plt.ylabel('Loss Function')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.savefig(self.out_dir + 'fitting_curve.png')


    def predict(self, model_dir=None):
        print('\n\nStart prediction of CAE...\n')
        if model_dir is not None:
            print('Loading model for prediction...\n')
            self.autoencoder = util.load_cae_model(self.model_name, model_dir)
        else:
            print('Using trained model for prediction...')

        # Get all validation data 
        val_data = [x[0] for x in self.validation_generator]
        val_data = np.array(val_data)
        print(val_data.shape)
        if 1 in self.patch_size: 
            val_data = val_data.reshape(val_data.shape[0]*val_data.shape[1], val_data.shape[2], val_data.shape[3], 1)
        else:
            val_data = val_data.reshape(val_data.shape[0]*val_data.shape[1], val_data.shape[2], val_data.shape[3], val_data.shape[4], 1)
        # Make predictions
        pred = self.autoencoder.predict(val_data, verbose=1, batch_size=self.batch_size)
        print(val_data.shape)
        
        # Calculate reconstruction error
        recon_error = tf.square(pred - val_data)

        # Convert to numpy and normalize
        recon_error = recon_error.numpy()

        # Calculate mse
        mse = np.mean(recon_error)
       
        if 1 not in self.patch_size:
            slice_num = np.random.randint(pred.shape[1]-1)
      
        nrows, ncols = 5, 3
        samples = [random.randint(1,len(val_data)-1) for i in range(nrows)]
        figsize = [15,15]
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, axi in enumerate(ax.flat):
            rowid = i // ncols
            colid = i % ncols
            ax[rowid,colid].axis('off') 
            if colid==0:
                axi.set_title('Original Patch', fontsize=15)
                if 1 not in self.patch_size:
                    axi.imshow(val_data[samples[rowid], slice_num, ...,0], cmap='gray')
                else:
                    axi.imshow(val_data[samples[rowid], ..., 0], cmap='gray')
            elif colid==1:
                axi.set_title('Reconstructed Patch', fontsize=15)
                if 1 not in self.patch_size:
                    axi.imshow(pred[samples[rowid], slice_num, ..., 0], cmap='gray')
                else:
                    axi.imshow(pred[samples[rowid], ..., 0], cmap='gray')
            else:
                axi.set_title('Reconstruction Error', fontsize=15)
                if 1 not in self.patch_size:
                    img=axi.imshow(recon_error[samples[rowid], slice_num, ..., 0], cmap='jet', norm=LogNorm(), vmin=0.00001, vmax=1)
                else:
                    img=axi.imshow(recon_error[samples[rowid], ..., 0], cmap='jet', norm=LogNorm(), vmin=0.00001, vmax=1)
                plt.colorbar(img, ax=axi)
            plt.tight_layout(True)
            plt.savefig(self.out_dir + 'error.png')  

        # Add parameters to dict and save it  
        info = {'patch_size': self.patch_size,
                'sampler_type': self.sampler_type,
                'max_patches': self.max_patches,
                'resample': self.resample,
                'clipping': self.clipping,
                'pixel_norm': self.pixel_norm,
                'patch_overlap': self.patch_overlap,
                'min_labeled_pixels': self.min_labeled_voxels,
                'label_probabilities': self.label_prob,
                'batch_size': self.batch_size,
                'test_size': self.test_size,
                'prepare_batches': self.prepare_batches
                }
        
        util.save_dict(info, self.out_dir, 'info.csv')
        util.save_dict({'mse': mse}, self.out_dir, 'evaluation.csv')

    
    def create_generators(self, prepare_batches, data_dir):
        # Generators
        if  prepare_batches:
            self.training_generator = DataGenerator(self.partition['train'], data_dir=data_dir, shuffle=True)
            self.validation_generator = DataGenerator(self.partition['validation'], data_dir=data_dir)
        else:
            self.training_generator = DataGenerator(self.partition['train'], data_dir=data_dir, shuffle=True, batch_size=self.batch_size)
            self.validation_generator = DataGenerator(self.partition['validation'], data_dir=data_dir, batch_size=self.batch_size)
