from utils import util, extract_patches
from keras.models import Model
import os
from keras.models import model_from_json
import numpy as np
from collections import Counter
from tqdm import tqdm
import time


class FeatureExtraction:
    def __init__(self, model_name, patch_size, patch_overlap, min_labeled_pixels, num_clusters, voxel_selection, max_patches, input_dir,  model_dir=None):
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_labeled_pixels = min_labeled_pixels
        self.num_clusters = num_clusters
        self.voxel_selection = voxel_selection
        self.max_patches = max_patches
        self.input_dir = input_dir
        self.model_dir = model_dir
        
        # Load encoder 
        self.autoencoder = util.load_cae_model(self.model_name, self.model_dir)
        self.encoder = Model(self.autoencoder.input, self.autoencoder.layers[-5].output)
        self.encoder.summary() 

        # Create out dirs if it doesn't exists 
        if not os.path.exists('evaluation/classification/features/'):
            os.makedirs('evaluation/classification/features/')

        self.out_dir = util.get_next_folder_name('evaluation/classification/features/', pattern='ex')
        os.makedirs(self.out_dir)


    
    def run(self, batch_size):
        # Get filenames 
        img_filenames = util.get_paths_from_tree(self.input_dir, 'imaging')
        seg_filenames = util.get_paths_from_tree(self.input_dir, 'segmentation')
        clus_filenames = util.get_paths_from_tree(self.input_dir, 'cluster')
        
        print('\n\nRunning feature extraction on samples.. \n\n')  
       
        features = []
        for i in tqdm(range(len(img_filenames))):
            # Append filename to new list to match input of create 2_patches 
            img = [img_filenames[i]]
            seg = [seg_filenames[i]]
            clus = [clus_filenames[i]]

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

            # Start feature extraction time
            tic = time.time()
            feature = dict().fromkeys(range(1,self.num_clusters+1), 0)
            print(feature)
            print('\nStart extracting features from image...')
            for i in range(len(img_patches)):
                # Find matching cluster based on center voxel
                if self.voxel_selection == 'center': 
                    cluster = int(clus_patches[i, x_center, y_center, z_center])
                
                # Find the matching cluster based on highest share 
                elif self.voxel_selection == 'highest_share':
                    patch = clus_patches[i].astype(int)
                    cluster = np.argmax(np.bincount(patch.flat))
                
                # Calculate and add max stds for each cluster
                if cluster != 0:
                    std = np.std(pred[i])
                    max_std = feature.get(cluster)
                    if std > max_std:
                        feature[cluster] = std
            
            # End clustering time
            toc = time.time()

            # Print output
            print("Image feature-extraction-time: " + str(round(toc - tic, 2)))

            # Add the max stds for the image 
            features.append(list(feature.values()))
            util.delete_tmp_files()
                
        # Save extracted features    
        util.save_fe_results(   features=features,
                                patch_size=self.patch_size,
                                patch_overlap=self.patch_overlap,
                                min_labeled_pixels=self.min_labeled_pixels,
                                num_clusters=self.num_clusters,
                                voxel_selection=self.voxel_selection,
                                input_dir=self.input_dir,
                                out_dir=self.out_dir    )   