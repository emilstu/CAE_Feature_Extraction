from numpy.core.shape_base import stack
from torchio.data import inference
from utils import util, extract_patches
from keras.models import Model
import os
import numpy as np
from tqdm import tqdm
import time
from utils.data_generator import DataGenerator
import shutil
from collections import defaultdict

class FeatureExtraction:
    def __init__(self, model_name, patch_size, min_labeled_voxels, cluster_selection, num_clusters, input_dir, model_dir, patch_overlap, resample, clipping, pixel_norm, encoded_layer_num, spn, save_patches):
        self.model_name = model_name
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.min_labeled_voxels = min_labeled_voxels
        self.num_clusters = num_clusters
        self.cluster_selection = cluster_selection
        self.resample = resample
        self.clipping=clipping
        self.pixel_norm=pixel_norm
        self.encoded_layer_num = encoded_layer_num
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.spn=spn
        self.save_patches = save_patches
        
        # Load encoder
        self.autoencoder = util.load_cae_model(self.model_name, self.model_dir)
        self.encoder = Model(self.autoencoder.input, self.autoencoder.layers[-self.encoded_layer_num].output)
        self.encoder.summary() 

        # Create out dirs if it doesn't exists 
        if not os.path.exists(f'{model_dir}features/'):
            os.makedirs(f'{model_dir}features/')
        self.out_dir = util.get_next_folder_name(f'{model_dir}/features/', pattern='ex')
        os.makedirs(f'{self.out_dir}method1/')
        os.makedirs(f'{self.out_dir}method2/')
        
        os.makedirs(f'{self.out_dir}global_std/')
        os.makedirs(f'{self.out_dir}min_loc_std/')
        os.makedirs(f'{self.out_dir}global_mean/')
        os.makedirs(f'{self.out_dir}max_loc_mean/')
        os.makedirs(f'{self.out_dir}min_loc_mean/')
        os.makedirs(f'{self.out_dir}pop_max_std/')
        os.makedirs(f'{self.out_dir}pop_min_std/')
    

    def run(self, batch_size):
        # Get filenames 
        img_filenames = util.get_paths_from_tree(self.input_dir, 'imaging')
        clus_filenames = util.get_paths_from_tree(self.input_dir, 'cluster')
        folder_names = util.get_sub_dirs(self.input_dir)

        img_filenames = img_filenames[self.spn:]
        clus_filenames = clus_filenames[self.spn:]
        folder_names = folder_names[self.spn:]

        print('Staring point: ', img_filenames[0])
        print('Staring point: ', clus_filenames[0])
        print('Staring point: ', folder_names[0])

        # Calculate center indices
        ps=np.array(self.patch_size, dtype=np.int16)
        ps=(ps[ps != 1])
        if 1 in self.patch_size:
            ic = int(ps[0]/2 - 1)
            jc = int(ps[1]/2 - 1)
            kc = 0
        else:
            ic = int(ps[0]/2 - 1)
            jc = int(ps[1]/2 - 1)
            kc = int(ps[2]/2 - 1)

        # Save info
        info = {'patch_size': self.patch_size,
                'patch_overlap': self.patch_overlap,
                'cluster_selection': self.cluster_selection,
                'num_clusters': self.num_clusters,
                'pixel_norm': self.pixel_norm,
                'clipping': self.clipping,
                'encoded_layer_num': self.encoded_layer_num,
                'min_labled_voxels': self.min_labeled_voxels,
                'voxel_resampling:': self.resample
                } 
        util.save_dict(info, self.out_dir, 'info.csv')
        
        save_dir = None

        #Clip pixels intensities and normalize data
        if self.clipping: img_filenames = util.clipping(img_filenames, clipping=self.clipping)
        if not self.clipping: self.img_filenames = util.normalize_data(img_filenames, norm=self.pixel_norm)

        print('\n\nRunning feature extraction on samples.. \n\n')
        pop_std=[]
        if self.save_patches: save_dir = util.create_fe_patch_dir(self.patch_size)
        for i in tqdm(range(len(img_filenames))):            
            # Append filename to new list to match input of create 2_patches 
            img = [img_filenames[i]]
            clus = [clus_filenames[i]]

            # Start feature extraction time
            tic = time.time()
            # Extract 3d patches from img
            image_data, cluster_data = extract_patches.patch_sampler(   img_filenames=img,
                                                                        labelmap_filenames=clus,
                                                                        patch_size=self.patch_size,
                                                                        sampler_type='grid',
                                                                        voxel_spacing=self.resample,
                                                                        patch_overlap=self.patch_overlap,
                                                                        pixel_norm=self.pixel_norm,
                                                                        out_dir=save_dir,    
                                                                        save_patches=self.save_patches,
                                                                        prepare_batches=True,
                                                                        batch_size=batch_size,
                                                                        inference = True,
                                                                        min_labeled_voxels=self.min_labeled_voxels)
                                    
            if self.save_patches:
                pop_std,patient_std,patient_mean,max_std_clus_enc=self.extraxt_encoding_features_from_saved_batches(    image_data, cluster_data,
                                                                                                                        ic, jc,kc, save_dir, batch_size, pop_std)
            
            else:
                # Make batch predictions
                pred = self.encoder.predict(image_data, verbose=1, batch_size=batch_size)
                pop_std,patient_std,patient_mean,max_std_clus_enc=self.extraxt_encoding_features(pred, cluster_data, ic, jc, kc, pop_std)
                     
            # End clustering time
            toc = time.time()
            print("Image feature-extraction-time: " + str(round(toc - tic, 2)))
            # Save features to disk 
            util.save_features(list(patient_std.get('glob_std')), folder_names[i], f'{self.out_dir}global_std/')
            util.save_features(list(patient_std.get('min_loc_std')), folder_names[i], f'{self.out_dir}min_loc_std/')
            util.save_features(list(patient_mean.get('glob_mean')), folder_names[i], f'{self.out_dir}global_mean/')
            util.save_features(list(patient_mean.get('max_loc_mean')), folder_names[i], f'{self.out_dir}max_loc_mean/')
            util.save_features(list(patient_mean.get('min_loc_mean')), folder_names[i], f'{self.out_dir}min_loc_mean/')

            util.save_features(list(patient_std.get('max_loc_std')), folder_names[i], f'{self.out_dir}method1/')
            util.save_features(list(max_std_clus_enc), folder_names[i], f'{self.out_dir}method2/')
        
        pop_max_std=np.amax(np.stack(pop_std, axis=0), axis=0)
        pop_min_std=np.amin(np.stack(pop_std, axis=0), axis=0)
        print('Population max std', pop_max_std.shape)
        
        util.save_features(pop_max_std, 'pop_max_std', f'{self.out_dir}pop_max_std/')
        util.save_features(pop_min_std, 'pop_min_std', f'{self.out_dir}pop_min_std/')
        
        
        if self.save_patches: shutil.rmtree(save_dir)

    def extraxt_encoding_features(self, pred, cluster_data, ic, jc, kc, pop_std):
        # Make batch predictions
        max_stds = dict().fromkeys(range(1,self.num_clusters+1), 0)
        feature_dict = defaultdict(list, { k:[] for k in range(1, self.num_clusters+1)})
        units=len(pred[0])
        print('Patient encodings shape', pred.shape)
        enc_global=[]
        for j in range(len(pred)):
            # Find matching cluster based on center voxel
            if self.cluster_selection == 'center': 
                cluster = int(cluster_data[j, ic, jc, kc])
            
            # Find the matching cluster based on highest share 
            elif self.cluster_selection == 'highest_share':
                cluster_patch = cluster_data[j].astype(int)
                cluster = np.argmax(np.bincount(cluster_patch.flat))
            if cluster != 0:
                feature_dict[cluster].append(pred[j])
                enc_global.append(pred[j])
                std = np.std(pred[j])
                max_std = max_stds.get(cluster)
                if std > max_std:
                    max_stds[cluster] = std
        
        enc_stds_local=[]
        enc_mean_local=[]

        # Find std for each feature for each cluster
        for value in feature_dict.values():
            if value: 
                stacked=np.stack(value, axis=0)
                enc_stds_local.append(np.std(stacked, axis=0))
                enc_mean_local.append(np.mean(stacked, axis=0))
            else: 
                enc_stds_local.append(np.zeros(shape=(units,)))
                enc_mean_local.append(np.zeros(shape=(units,)))

        enc_stds_local=np.stack(enc_stds_local, axis=0)
        enc_global = np.stack(enc_global, axis=0)
        enc_mean_local = np.stack(enc_mean_local, axis=0)

        # Find min/max local stds
        max_loc_std=np.amax(enc_stds_local, axis=0)
        min_loc_std=np.amin(enc_stds_local, axis=0)
        pop_std.append(np.amax(enc_stds_local, axis=0))
        
        #min_loc_std=np.amin(enc_mean_local, axis=0)
        #max_loc_std=np.mean(enc_mean_local, axis=0)

        # Find min/max local mean 
        max_loc_mean=np.amax(enc_mean_local, axis=0)
        min_loc_mean=np.amin(enc_mean_local, axis=0)
        
        # Find global mean and std 
        glob_mean = np.mean(enc_global, axis=0)
        glob_std  = np.std(enc_global, axis=0)

        print('Max local std: ', max_loc_std.shape)
        print('Min local std: ', min_loc_std.shape)
        print('Max local mean: ', min_loc_mean.shape)
        print('Min local mean Shape: ', max_loc_mean.shape)
        print('Global std Shape: ', glob_std.shape)
        print('Global mean Shape: ', glob_mean.shape)
        
        patient_std={'glob_std': glob_std,
                    'max_loc_std': max_loc_std,
                    'min_loc_std': min_loc_std}
        patient_mean={'glob_mean': glob_mean,
                    'max_loc_mean': max_loc_mean,
                    'min_loc_mean': min_loc_mean}

        max_std_clus_enc=max_stds.values()
        
        return pop_std, patient_std, patient_mean, max_std_clus_enc

    
    def extraxt_encoding_features_from_saved_batches(self, image_ids, cluster_ids, ic, jc, kc, save_dir, batch_size, pop_std):
        encodings=[]    
        clusters=[]
        for j in range(len(image_ids)):
            # Add partition to dict and save
            partition = dict()
            partition['image'] = [image_ids[j]]
            partition['cluster'] = [cluster_ids[j]]

            print(partition)
            # Create generators
            image_generator = DataGenerator(partition['image'], data_dir=save_dir, shuffle=False)
            cluster_generator = DataGenerator(partition['cluster'], data_dir=save_dir, shuffle=False)
            # Load all data from generator
            cluster_batch=np.array([x[0] for x in cluster_generator]).squeeze(axis=0)
            encoded=self.encoder.predict(image_generator, verbose=1, batch_size=batch_size)
            if self.cluster_selection=='center':
                for k in range(len(cluster_batch)):
                    clusters.append(int(cluster_batch[k, ic, jc, kc]))
                    encodings.append(encoded[k])
            else:
                for k in range(len(cluster_batch)):
                    cluster_patch = cluster_batch[k].astype(int)
                    clusters.append(np.argmax(np.bincount(cluster_patch.flat)))
                    encodings.append(encoded[k])

        encodings=np.array(encodings)
        clusters=np.array(clusters,dtype=np.int16)

        # Make batch predictions
        max_stds = dict().fromkeys(range(1,self.num_clusters+1), 0)
        feature_dict = defaultdict(list, { k:[] for k in range(1, self.num_clusters+1)})
        units=len(encodings[0])
        for l in range(len(encodings)):
            cluster=clusters[l]
            encoding=encodings[l]
            # Calculate and add max stds for each cluster
            if cluster != 0:
                feature_dict[cluster].append(encoding)
                std = np.std(encoding)
                max_std = max_stds.get(cluster)
                if std > max_std:
                    max_stds[cluster] = std
        
        enc_stds_local=[]
        enc_mean_local=[]
        # Find std for each feature for each cluster
        for value in feature_dict.values():
            if value: 
                stacked=np.stack(value, axis=0)
                enc_stds_local.append(np.std(stacked, axis=0))
                enc_mean_local.append(np.mean(stacked, axis=0))
            else: 
                enc_stds_local.append(np.zeros(shape=(units,)))
                enc_mean_local.append(np.zeros(shape=(units,)))
        
        
        enc_stds_local=np.stack(enc_stds_local, axis=0)
        enc_global = np.stack(encodings, axis=0)
        enc_mean_local = np.stack(enc_mean_local, axis=0)
        
        # Find min/max local stds
        max_loc_std=np.amax(enc_stds_local, axis=0)
        min_loc_std=np.amin(enc_stds_local, axis=0)
        pop_std.append(np.amax(enc_stds_local, axis=0))
        
        # Find min/max local mean 
        max_loc_mean=np.amax(enc_mean_local, axis=0)
        min_loc_mean=np.amin(enc_mean_local, axis=0)
        
        # Find global mean and std 
        glob_mean = np.mean(enc_global, axis=0)
        glob_std  = np.std(enc_global, axis=0)

        print('Max local std Shape: ', max_loc_std.shape)
        print('Min local std Shape: ', min_loc_std.shape)
        print('Max local mean Shape: ', min_loc_mean.shape)
        print('Min local mean Shape: ', max_loc_mean.shape)
        print('Global std Shape: ', glob_std.shape)
        print('Global mean Shape: ', glob_mean.shape)
        
        patient_std={'glob_std': glob_std,
                    'max_loc_std': max_loc_std,
                    'min_loc_std': min_loc_std}
        patient_mean={'glob_mean': glob_mean,
                    'max_loc_mean': max_loc_mean,
                    'min_loc_mean': min_loc_mean}

        max_std_clus_enc=max_stds.values()
        
        return pop_std, patient_std, patient_mean, max_std_clus_enc