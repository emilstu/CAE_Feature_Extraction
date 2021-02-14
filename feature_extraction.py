from utils import util, extract_patches
from keras.models import Model
import os
from keras.models import model_from_json
import numpy as np
from collections import Counter
import statistics


class FeatureExtraction:
    def __init__(self, model_name, patch_size, patch_overlap, min_labeled_pixels, num_clusters, voxel_selection, max_patches, img_dir, seg_dir, clus_dir, out_dir, model_dir=None):
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_labeled_pixels = min_labeled_pixels
        self.num_clusters = num_clusters
        self.voxel_selection = voxel_selection
        self.max_patches = max_patches

        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.clus_dir = clus_dir
        self.out_dir = out_dir
        self.model_dir = model_dir
        
        self.features = []
        self.img_patch_list = []
        self.clus_patch_list = []

        self.img_filenames = util.get_nifti_filenames(self.img_dir)
        self.clus_filenames = util.get_nifti_filenames(self.clus_dir)
        self.seg_filenames = util.get_nifti_filenames(self.seg_dir)

        # Load encoder 
        self.autoencoder = util.load_json_model(self.model_name, self.model_dir)
        self.encoder = Model(self.autoencoder.input, self.autoencoder.layers[-5].output)
        self.encoder.summary() 

        # Create out directory if it doesn't exists 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


    
    def run(self, batch_size):
        for i in range(len(self.img_filenames)):
            # Append filename to new list to match input of create 2_patches 
            img = [self.img_filenames[i]]
            seg = [self.seg_filenames[i]]
            clus = [self.clus_filenames[i]]

            if self.patch_size[0] == 1:
                # Extract 2d patches from img
                img_patch_list, _, clus_patch_list = extract_patches.extract_2d_patches(    img_filenames=img,
                                                                                            clus_filenames=clus,
                                                                                            patch_size=self.patch_size,
                                                                                            patch_overlap=self.patch_overlap,
                                                                                            min_labeled_pixels=self.min_labeled_pixels  )

                # Save sample lists to disk 
                util.save_sample_list(img_patch_list, name=f'img_patches_{i}', out_dir=self.out_dir)
                util.save_sample_list(clus_patch_list, name=f'clus_patches_{i}', out_dir=self.out_dir)
                
                # Load patches from disk
                img_patches = util.load_patches(img_patch_list)
                clus_patches = util.load_patches(clus_patch_list)
                
                # Normalize data
                img_patches = util.normalize_data(img_patches)

                # Reshape data from (num_samples, 1, 48, 48) to (num_samples, 48, 48, 1)
                img_patches = util.reshape_data_2d(img_patches, self.patch_size)
                clus_patches = util.reshape_data_2d(clus_patches, self.patch_size)
            
            else:
                # Extract 3d patches from img
                img_patches, clus_patches = extract_patches.extract_3d_patches( img_filenames=img,
                                                                                seg_filenames=seg,
                                                                                clus_filenames=clus,
                                                                                patch_size=self.patch_size,
                                                                                max_patches=self.max_patches    )

                # Convert to numpy 
                img_patches = np.asarray(img_patches)
                clus_patches = np.asarray(clus_patches)


                # Normalize data 
                img_patches = util.normalize_data(img_patches)


            # Make predictions 
            pred = self.encoder.predict(img_patches, verbose=1, batch_size=batch_size)

            # Calculate center index
            x_center = 0
            y_center = 0
            z_center = 0
            
            if self.patch_size[0] != 1:
                x_center = int(self.patch_size[0]/2 - 1)
                y_center = int(self.patch_size[1]/2 - 1)
                z_center = int(self.patch_size[2]/2 - 1)
            else:
                x_center = int(self.patch_size[1]/2 - 1)
                y_center = int(self.patch_size[2]/2 - 1)

           
            feature = [[] for i in range(self.num_clusters)]
            for i in range(len(img_patches)):
                if self.voxel_selection == 'center':
                    # Find matching cluster based on center voxel 
                    cluster = int(clus_patches[i, x_center, y_center, z_center])
                
                    # Add the extracted feature to the feature list
                    if cluster != 0:
                        std = np.std(pred[i]) 
                        feature[cluster-1].append(std)
                elif self.voxel_selection == 'highest_share':
                    # Find the matching cluster pased on highest share 
                    patch = clus_patches[i].astype(int)
                    cluster = np.argmax(np.bincount(patch.flat))
                    if cluster != 0:
                        std = np.std(pred[i])
                        feature[cluster-1].append(std)
                
            # Find maximum value for each cluster 
            max_stds = []
            for i in range (len(feature)):
                val = feature[i]
                if val:
                    max_stds.append(max(val))
                else:
                    max_stds.append(0)
            
            self.features.append(max_stds)
                
                

            
            
        # Save results
        util.save_fe_results(   features=self.features,
                                patch_size=self.patch_size,
                                patch_overlap=self.patch_overlap,
                                min_labeled_pixels=self.min_labeled_pixels,
                                num_clusters=self.num_clusters,
                                voxel_selection=self.voxel_selection,
                                out_dir=self.out_dir                            )
        
        
            