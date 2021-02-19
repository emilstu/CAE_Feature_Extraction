from tensorflow.python.ops.gen_batch_ops import batch
from utils import util, extract_patches
from CAE.cae_2d import CAE_2D
from CAE.cae_3d import CAE_3D
import os
import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
import random

class CAE:
    def __init__(self, patch_size, min_labeled_pixels, test_size, input_dir, max_patches=None, patch_overlap=None, model_dir=None):
        if patch_size[0] == 1:
            if patch_overlap is None:
                raise Exception('\nException: patch-overlap must me specified for 2D autoencoder \n') 
            
            # Create out dirs if it doesn't exists
            if not os.path.exists('evaluation/CAE/2D/'):
                os.makedirs('evaluation/CAE/2D/')

            model_name = 'model_2D'
            out_dir = util.get_next_folder_name('evaluation/CAE/2D/', pattern='ex')
            os.makedirs(out_dir)

            self.cae = CAE_2D(  model_name=model_name,
                                patch_size=patch_size,
                                patch_overlap=patch_overlap,
                                min_labeled_pixels=min_labeled_pixels,
                                test_size=test_size,
                                input_dir=input_dir,  
                                out_dir=out_dir )
    
        else:
            
            # Create out dirs if it doesn't exists
            if not os.path.exists('evaluation/CAE/3D/'):
                os.makedirs('evaluation/CAE/3D/')

            model_name = 'model_3D'
            out_dir = util.get_next_folder_name('evaluation/CAE/3D/', pattern='ex')
            os.makedirs(out_dir)

            self.cae = CAE_3D(  model_name=model_name,
                                patch_size=patch_size,
                                min_labeled_pixels=min_labeled_pixels,
                                max_patches=max_patches,
                                test_size=test_size,
                                input_dir=input_dir,
                                out_dir=out_dir  )
            
  
    def train(self, batch_size, epochs, batches_per_epoch, delete_patches=False):
        print('\n\nStart training of CAE...\n')
        self.cae.extract_patches()
        self.cae.train(batch_size, epochs, batches_per_epoch)

        if delete_patches:
            self.cae.delete_patches()

    def predict(self, batch_size, model_dir=None, delete_patches=False):
        print('\n\nStart prediction of CAE...\n')
        if model_dir is not None:
            print('Loading model for prediction...\n')
            self.cae.load(model_dir)
        else:
            print('Using trained model for prediction...')
        
        self.cae.predict(batch_size)

        if delete_patches:
            self.cae.delete_patches()

    



        

                


            
