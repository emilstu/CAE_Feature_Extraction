from utils import util, extract_patches
from keras.models import Model
import os
from keras.models import model_from_json
import numpy as np
from tqdm import tqdm
import time


class FeatureExtraction:
    def __init__(self, model_name, patch_size, cluster_selection, num_clusters, input_dir, model_dir, patch_overlap=None, min_labeled_pixels=None, max_patches=None):
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_labeled_pixels = min_labeled_pixels
        self.num_clusters = num_clusters
        self.cluster_selection = cluster_selection
        self.max_patches = max_patches
        self.input_dir = input_dir
        self.model_dir = model_dir
        
        # Load encoder 
        self.autoencoder = util.load_cae_model(self.model_name, self.model_dir)
        self.encoder = Model(self.autoencoder.input, self.autoencoder.layers[-6].output)
        self.encoder.summary() 

        # Create out dirs if it doesn't exists 
        if not os.path.exists('evaluation/classification/features/'):
            os.makedirs('evaluation/classification/features/')

        self.out_dir = util.get_next_folder_name('evaluation/classification/features/', pattern='ex')
        os.makedirs(self.out_dir)


    
    def run(self, batch_size):
        # Get filenames 
        img_filenames = util.get_paths_from_tree(self.input_dir, 'imaging')
        clus_filenames = util.get_paths_from_tree(self.input_dir, 'cluster')
        folder_names = util.get_sub_dirs(self.input_dir)
        
        print('\n\nRunning feature extraction on samples.. \n\n')  
        for i in tqdm(range(len(img_filenames))):
            # Append filename to new list to match input of create 2_patches 
            img = [img_filenames[i]]
            clus = [clus_filenames[i]]

            if self.patch_size[0] == 1:
                # Extract 2d patches from img
                img_patches, clus_patches = extract_patches.extract_2d_patches( img_filenames=img,
                                                                                labelmap_filenames=clus,
                                                                                patch_size=self.patch_size,
                                                                                patch_overlap=self.patch_overlap,
                                                                                min_labeled_pixels=self.min_labeled_pixels  )
                # Normalize data
                img_patches = util.normalize_data(img_patches)

                # Save info 
                info = {'patch_size': self.patch_size,
                        'patch_overlap': self.patch_overlap,
                        'min_labeled_pixels': self.min_labeled_pixels,
                        'cluster_selection': self.cluster_selection,
                        'num_clusters': self.num_clusters
                        }
                util.save_dict(info, self.out_dir, 'info.csv')

            else:
                # Extract 3d patches from img
                img_patches, clus_patches = extract_patches.extract_3d_patches( img_filenames=img,
                                                                                labelmap_filenames=clus,
                                                                                patch_size=self.patch_size,
                                                                                max_patches=self.max_patches,
                                                                                extract_labelmap_patches=True   )

                # Normalize data 
                img_patches = util.normalize_data(img_patches)

                # Save info
                info = {'patch_size': self.patch_size,
                        'max_patches': self.max_patches,
                        'cluster_selection': self.cluster_selection,
                        'num_clusters': self.num_clusters
                        } 
                util.save_dict(info, self.out_dir, 'info.csv')

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
            print('\nStart extracting features from image...')
            for j in range(len(img_patches)):
                # Find matching cluster based on center voxel
                if self.cluster_selection == 'center': 
                    cluster = int(clus_patches[j, x_center, y_center, z_center])
                
                # Find the matching cluster based on highest share 
                elif self.cluster_selection == 'highest_share':
                    patch = clus_patches[j].astype(int)
                    cluster = np.argmax(np.bincount(patch.flat))
                
                # Calculate and add max stds for each cluster
                if cluster != 0:
                    std = np.std(pred[j])
                    max_std = feature.get(cluster)
                    if std > max_std:
                        feature[cluster] = std
            
            # End clustering time
            toc = time.time()
            print("Image feature-extraction-time: " + str(round(toc - tic, 2)))

            # Save features to disk 
            util.save_features(list(feature.values()), folder_names[i], self.out_dir)
        