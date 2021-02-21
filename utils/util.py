import numpy as np
import glob
import nibabel as nib
import torch
import torchio as tio
import random
from keras.models import model_from_json 
import pickle
import os
import csv
from sklearn.utils import check_array, check_random_state
import numbers
from numpy.lib.stride_tricks import as_strided
import shutil
from csv import reader
import copy
from skimage import morphology

def get_nifti_filepaths(dir):
    filepaths=glob.glob(dir + '*.nii.gz')
    filepaths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return filepaths

def get_nifti_filenames(dir, format=False):
    filepaths=glob.glob(dir + '*.nii.gz')
    filepaths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    filenames = []
    for path in filepaths:
        name = path.split('/')[-1]
        if not format:
            name = name.split('.')[0]
        filenames.append(name)
    return filenames
    
def get_sub_dirs(dir):
    dirlist = [ item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item)) ]
    dirlist.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    return dirlist

def get_next_folder_name(dir, pattern):
    next_num = 1
    dirlist = [ item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item)) ]
    if dirlist:
        dirlist.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        next_num = int(dirlist[-1].split(pattern)[1]) + 1
    next_folder_name = f'{dir}{pattern}{next_num}/'
    return next_folder_name


def get_paths_from_tree(dir, type='imaging'):
    dirlist = [ item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item)) ]
    if dirlist:
        dirlist.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    else:
        raise Exception(f'\nError: no images to be loaded from {dir}\n')

    file_paths = []
    for i in range(len(dirlist)):
        file_paths.append(f'{dir}{dirlist[i]}/{type}.nii.gz')
    
    return file_paths



def save_sample_list(filenames, name, out_dir):
    filenames_save = np.asarray(filenames)
    with open(out_dir + name, 'wb') as fp:
        pickle.dump(filenames_save, fp)
    print(f'Saved {name} to disk')

    
def load_sample_list(name, dir):
    with open(dir + name, 'rb') as fp:
        sample_list = pickle.load(fp)
    
    print(f'Loaded {name} from disk')
    return sample_list
    

def load_patches(filenames):
    
    image_array = []
    # Loop over images
    for filename in filenames:
        img = nib.load(filename)
        img_data = img.get_data()
        image_array.append(img_data)
    
    image_array = np.array(image_array)

    return image_array


def normalize_data(data):
    new_data = data.astype('float32') / np.max(data)
    return new_data

def reshape_data_2d(data, patch_size):
    # Reshape data to fit into CAE
    new_data = np.reshape(data, (len(data), patch_size[1], patch_size[2], 1))
    return new_data


def load_cae_model(model_name, model_dir):
    # load json and create model
    json_file = open(f'{model_dir}{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
        
    # load weights into new model
    model.load_weights(f'{model_dir}{model_name}.h5')
    print("Loaded model from disk")

    return model

def save_cae_model(model, model_name, out_dir):
    model_json = model.to_json()
    with open(f'{out_dir}{model_name}.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(f'{out_dir}{model_name}.h5')
    print('Saved model to disk')


def save_svm_model(model, model_name, out_dir):
    filename = f'{out_dir}{model_name}.sav'
    pickle.dump(model, open(filename, 'wb'))


def load_svm_model(model_name, model_dir):
    loaded_model = pickle.load(open(f'{model_dir}{model_name}', 'rb'))
    print(loaded_model)


def save_svm_results(accuracy, precision, recall, out_dir):
    results = {'accuracy': accuracy,
            'precision': precision,
            'recall': recall
            }
    # Save dicts
    with open(f'{out_dir}results.csv', 'w') as f:
        for key in results.keys():
            f.write("%s: %s\n"%(key,results[key]))


def save_fe_results(features, patch_size, patch_overlap, min_labeled_pixels, num_clusters, voxel_selection, input_dir, out_dir): 
    # Add info to dict 
    info = {'patch_size': patch_size,
            'patch_overlap': patch_overlap,
            'min_labeled_pixels': min_labeled_pixels,
            'voxel_selection': voxel_selection,
            'num_clusters': num_clusters
            }
    
    # Add results to dict
    folder_names = get_sub_dirs(input_dir)
    results = {}
    for i in range(len(features)):
        results[folder_names[i]] = features[i]
    
    # Save dicts
    with open(f'{out_dir}info.csv', 'w') as f:
        for key in info.keys():
            f.write("%s: %s\n"%(key,info[key]))

    with open(f'{out_dir}result.csv', 'w') as f:
        for key in results.keys():
            f.write("%s: %s\n"%(key,results[key]))
    
    #save_sample_list(features, features, out_dir)


def load_features(feature_dir):
    print('Loading features...')
    features = []
    with open(f'{feature_dir}result.csv', 'r') as f:
        for row in f:
            feature = row.split(' ', 1)[1]
            feature = feature.strip(']\n[').split(', ')
            features.append(feature)
    print('Loaded features')
    # Convert to numpy 
    features = np.asarray(features, dtype=np.float64)
    return features

def ffr_values_to_target_list(target_dir, ffr_boundary=0.85):
    with open(f'{target_dir}ffr_values.txt') as f:
        values = f.readlines()
    
    #values = [x.strip() for x in values]
    target = []
    
    for value in values:
        if float(value) >= ffr_boundary:
            target.append(1)
        else:
            target.append(0)

    return np.array(target)


def delete_tmp_files():
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')


