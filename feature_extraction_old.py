from keras.models import Model
import util
import os
from keras.models import model_from_json
import numpy as np
from collections import Counter


class FeatureExtraction:
    def __init__(self, model_name, patch_size, patch_overlap, min_labeled_pixels, load_patches, num_clusters, voxel_selection, img_dir, seg_dir, clus_dir, out_dir, model_dir=None):
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_labeled_pixels = min_labeled_pixels
        self.load_patches = load_patches
        self.num_clusters = num_clusters
        self.voxel_selection = voxel_selection
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.clus_dir = clus_dir
        self.out_dir = out_dir
        self.model_dir = model_dir
        
        self.features = [[] for i in range(num_clusters)]
        self.autoencoder = Model()
        self.encoder = Model()
        self.img_patch_list = []
        self.seg_patch_list = []
        self.clus_patch_list = []

        # Create out directory if it doesn't exists 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if self.load_patches:
            self.load_patch_list_from_disk()
            
        else: 
            self.create_patches()


    def load_patch_list_from_disk(self):
        self.img_patch_list = util.load_sample_list('img_patches', self.out_dir)
        self.seg_patch_list = util.load_sample_list('seg_patches', self.out_dir)
        self.clus_patch_list = util.load_sample_list('clus_patches', self.out_dir)


    def create_patches(self):    
        # Create patches (Change create_2d_patches so it takes in outpath?)
        img_patch_list, seg_patch_list, clus_patch_list = util.create_2d_patches(   self.img_dir,
                                                                                    self.seg_dir,
                                                                                    self.clus_dir,
                                                                                    patch_size=self.patch_size,
                                                                                    patch_overlap=self.patch_overlap,
                                                                                    min_labeled_pixels=self.min_labeled_pixels   )
    
        # Save sample lists to disk 
        util.save_sample_list(img_patch_list, name='img_patches', out_dir=self.out_dir)
        util.save_sample_list(seg_patch_list, name='seg_patches', out_dir=self.out_dir)
        util.save_sample_list(clus_patch_list, name='clus_patches', out_dir=self.out_dir)

    
    def run(self, batch_size):
        # Load patches from disk
        img_patches = util.load_patches(self.img_patch_list)
        seg_patches = util.load_patches(self.seg_patch_list)
        clus_patches = util.load_patches(self.clus_patch_list)
       
        # Load model and weights 
        self.autoencoder = util.load_json_model(self.model_name, self.model_dir)

        # Get encoder for feature extraction 
        encoder = Model(self.autoencoder.input, self.autoencoder.layers[-5].output)
        encoder.summary()  
    

        # Normalize and reshape from (num_samples, 1, 48, 48) to (num_samples, 48, 48, 1)
        img_patches = util.preprocess_data(img_patches, self.patch_size, normalize=True)
        seg_patches = util.preprocess_data(seg_patches, self.patch_size)
        clus_patches = util.preprocess_data(clus_patches, self.patch_size)

        # Make predictions 
        pred = encoder.predict(img_patches, verbose=1, batch_size=batch_size)

        # Calculate center index
        x_center = int(self.patch_size[1]/2 - 1)
        y_center = int(self.patch_size[2]/2 - 1)

        for i in range(len(pred)):
            if self.voxel_selection == 'center':
                # Find matching cluster based on center voxel 
                cluster = int(clus_patches[i, y_center, x_center, 0])
            
                # Add the extracted feature to the feature list
                if cluster != 0: 
                    self.features[cluster-1].append(pred[i])
            elif self.voxel_selection == 'highest_share':
                # Find the matching cluster pased on highest share 
                patch = clus_patches[i, ..., 0].astype(int)
                cluster = np.argmax(np.bincount(patch.flat))
                if cluster != 0:
                    self.features[cluster-1].append(pred[i])
                

        max_stds = np.zeros(self.num_clusters)
        
        # For each cluster 
        for i in range(len(self.features)):
          
            # For each feature in the cluster
            for j in range(len(self.features[i])):
                # Calculate the standard derivation for each feature 
                self.features[i][j] = np.std(self.features[i][j])
            
            # Get the maximum standard derivation for the cluster
            if self.features[i]:
                self.features[i] = np.max(self.features[i])
            else:
                self.features[i] = 0.0       

        # Transform to numpy array 
        #self.features = np.asarray(self.features)
        
        # Save results
        util.save_fe_results(   features=self.features,
                                patch_size=self.patch_size,
                                patch_overlap=self.patch_overlap,
                                min_labeled_pixels=self.min_labeled_pixels,
                                num_clusters=self.num_clusters,
                                voxel_selection=self.voxel_selection,
                                out_dir=self.out_dir                            )
        
        
            