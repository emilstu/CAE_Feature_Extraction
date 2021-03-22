from torchio.data import inference
from utils import util, extract_patches
from keras.models import Model
import os
import numpy as np
from tqdm import tqdm
import time
from CAE.data_generator import DataGenerator
import shutil

class FeatureExtraction:
    def __init__(self, model_name, patch_size, cluster_selection, num_clusters, input_dir, model_dir, patch_overlap, resample, encoded_layer_num):
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.num_clusters = num_clusters
        self.cluster_selection = cluster_selection
        self.resample = resample
        self.encoded_layer_num = encoded_layer_num
        self.input_dir = input_dir
        self.model_dir = model_dir
        
        # Load encoder
        self.autoencoder = util.load_cae_model(self.model_name, self.model_dir)
        self.encoder = Model(self.autoencoder.input, self.autoencoder.layers[-self.encoded_layer_num].output)
        self.encoder.summary() 

        # Create out dirs if it doesn't exists 
        if not os.path.exists('evaluation/classification/features/'):
            os.makedirs('evaluation/classification/features/')

        self.out_dir = util.get_next_folder_name('evaluation/classification/features/', pattern='ex')
        os.makedirs(f'{self.out_dir}results')
    
    def run(self, batch_size):
        # Get filenames 
        img_filenames = util.get_paths_from_tree(self.input_dir, 'imaging')
        clus_filenames = util.get_paths_from_tree(self.input_dir, 'cluster')
        folder_names = util.get_sub_dirs(self.input_dir)
        
        print('\n\nRunning feature extraction on samples.. \n\n')  
        for i in tqdm(range(len(img_filenames))):
            # Create out dir for patches
            save_dir=util.create_fe_patch_dir(self.patch_size)
            
            # Append filename to new list to match input of create 2_patches 
            img = [img_filenames[i]]
            clus = [clus_filenames[i]]

    
            # Extract 3d patches from img
            img_ids, label_ids = extract_patches.patch_sampler( img_filenames=img,
                                                                labelmap_filenames=clus,
                                                                patch_size=self.patch_size,
                                                                out_dir=save_dir,
                                                                sampler_type='grid',
                                                                voxel_spacing=self.resample,
                                                                patch_overlap=self.patch_overlap,    
                                                                save_patches=True,
                                                                inference=True   )
            # Save info
            info = {'patch_size': self.patch_size,
                    'cluster_selection': self.cluster_selection,
                    'num_clusters': self.num_clusters,
                    'patch_overlap': self.patch_overlap
                    } 
            util.save_dict(info, self.out_dir, 'info.csv')
            
            # Add partition to dict and save
            partition = dict()
            partition['image'] = img_ids
            partition['cluster'] = label_ids

            # Create generators
            image_generator = DataGenerator(partition['image'], data_dir=save_dir, shuffle=False, batch_size=batch_size)
            cluster_generator = DataGenerator(partition['cluster'], data_dir=save_dir, shuffle=False, batch_size=batch_size)

            # Load all data from generator
            image_data = [x[0] for x in image_generator]
            cluster_data = [x[0] for x in cluster_generator] 
            image_data = np.array(image_data)
            cluster_data = np.array(cluster_data)
            if self.patch_size[0] == 1: 
                image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_data.shape[2], image_data.shape[3], 1)
                cluster_data = cluster_data.reshape(cluster_data.shape[0]*cluster_data.shape[1], cluster_data.shape[2], cluster_data.shape[3], 1)
            else:
                image_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_data.shape[2], image_data.shape[3], image_data.shape[4], 1)
                cluster_data = cluster_data.reshape(cluster_data.shape[0]*cluster_data.shape[1], cluster_data.shape[2], cluster_data.shape[3], cluster_data.shape[4], 1)
            print(image_data.shape)
            # Make predictions 
            pred = self.encoder.predict(image_data, verbose=1, batch_size=batch_size)

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
            for j in range(len(image_data)):
                # Find matching cluster based on center voxel
                if self.cluster_selection == 'center': 
                    cluster = int(cluster_data[j, x_center, y_center, z_center])
                
                # Find the matching cluster based on highest share 
                elif self.cluster_selection == 'highest_share':
                    cluster_patch = cluster_data[j].astype(int)
                    cluster = np.argmax(np.bincount(cluster_patch.flat))
                
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

        # Delete last img patches
        shutil.rmtree(save_dir)