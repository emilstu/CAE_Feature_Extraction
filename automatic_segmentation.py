from operator import imod
import tensorflow as tf
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.processing.subfunctions import Normalization, Resampling
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, \
                                          dice_crossentropy, tversky_loss
from miscnn.evaluation.cross_validation import load_disk2fold
import os
from utils import util
import shutil
import nibabel as nib
from skimage import morphology
import copy
import numpy as np
from tqdm import tqdm

class AutomaticSegmentation:
    def __init__(self, model_name, patch_size, input_dir, model_dir):
        self.model_name = model_name
        self.patch_size = patch_size
        self.predictions = []
        self.model_dir = model_dir
        self.input_dir = input_dir

    
    def run(self):
       
        # Initialize Data IO Interface for NIfTI data
        interface = NIFTI_interface(channels=1, classes=2)

        # Create Data IO object to load and write samples in the file structure
        data_io = Data_IO(interface, input_path=self.input_dir, delete_batchDir=False)


        # Access all available samples in our file structure
        sample_list = data_io.get_indiceslist()
        sample_list.sort()   

        # Create a resampling Subfunction to voxel spacing 1.58 x 1.58 x 2.70
        sf_resample = Resampling((1.58, 1.58, 2.70))

        # Create a pixel value normalization Subfunction for z-score scaling
        sf_zscore = Normalization(mode="z-score")

        # Create a pixel value normalization Subfunction to scale between 0-255
        sf_normalize = Normalization(mode="grayscale")

        # Assemble Subfunction classes into a list
        sf = [sf_normalize, sf_resample, sf_zscore]


        # Create and configure the Preprocessor class
        pp = Preprocessor(  data_io, batch_size=2, subfunctions=sf,
                            prepare_subfunctions=True, prepare_batches=False,
                            analysis="patchwise-crop", patch_shape=(160, 160, 80))
        
        # Adjust the patch overlap for predictions
        pp.patchwise_overlap = (80, 80, 30)
        
        # Initialize the Architecture
        unet_standard = Architecture(   depth=4, activation="softmax",
                                        batch_normalization=True)

        # Create the Neural Network model
        model = Neural_Network( preprocessor=pp, architecture=unet_standard,
                                loss=tversky_crossentropy,
                                metrics=[tversky_loss, dice_soft, dice_crossentropy],
                                batch_queue_size=3, workers=1, learninig_rate=0.001)


        # Load best model weights during fitting
        model.load(f'{self.model_dir}{self.model_name}.hdf5')

        # Obtain training and validation data set ----- CHANGE BASED ON PRED/TRAIN
        images ,_ = load_disk2fold(f'{self.input_dir}sample_list.json')

 
        print('\n\nRunning automatic segmentation on samples...\n')
        print(f'Segmenting images: {images}')

        # Compute predictions
        self.predictions = model.predict(images)

        # Delete folder created by miscnn
        shutil.rmtree('batches/')

    def run_postprocessing(self):
        print('Post-processing of segmented images...')
        print('Clipping images... ')
    
        filepaths = util.get_nifti_filepaths('predictions/')
        filenames = util.get_sub_dirs(self.input_dir)

        for i in tqdm(range(len(filepaths))): 
            self.remove_small_objects(filepaths[i], filenames[i], self.input_dir)
        
        # Delete folder created by miscnn
        shutil.rmtree('predictions/')


    def remove_small_objects(self, filepath, filename, out_dir):
        img = nib.load(filepath)
        # to be extra sure of not overwriting data:
        binary = copy.copy(img.get_data())
        hd = img.header

        binary[binary>0] = float(1)
        labels = morphology.label(binary)
        labels_num = [len(labels[labels==each]) for each in np.unique(labels)]
        rank = np.argsort(np.argsort(labels_num))
        index = list(rank).index(len(rank)-2)
        new_img = copy.copy(img.get_data())
        new_img[labels!=index] = float(0)

        # update data type:
        new_dtype = np.float32  # for example to cast to int8.
        new_img = new_img.astype(new_dtype)
        img.set_data_dtype(new_dtype)
        
        # if nifty1
        if hd['sizeof_hdr'] == 348:
            new_image = nib.Nifti1Image(new_img, img.affine, header=hd)
        # if nifty2
        elif hd['sizeof_hdr'] == 540:
            new_image = nib.Nifti2Image(new_img, img.affine, header=hd)
        else:
            raise IOError('Input image header problem')

        out_path = f'{out_dir}{filename}/segmentation.nii.gz'

        print(out_path)
        nib.save(new_image, out_path)
