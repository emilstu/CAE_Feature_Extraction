import numpy as np
import glob
import nibabel as nib
from nibabel import processing
from keras.models import model_from_json 
import pickle
import os
from numpy.core.fromnumeric import partition
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
import shutil
from collections import Counter
from collections import defaultdict
import json
import csv
from scipy.stats import stats

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


def save_features(features, patient, out_dir): 
    # Append feature to file
    with open(f'{out_dir}{patient}.csv', 'w', newline='') as f:
        wr=csv.writer(f)
        wr.writerow(features)
    
def save_dict(dict, out_dir, filename):
    with open(f'{out_dir}{filename}', 'w') as f:
        for key in dict.keys():
            f.write("%s: %s\n"%(key,dict[key]))

def create_cae_patch_dir(prepare_batches, patch_size):
        save_dir=get_cae_patch_dir(prepare_batches, patch_size)
        # Remove dir if it already exists 
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        return save_dir

def create_fe_patch_dir(patch_size):
    save_dir=get_fe_patch_dir(patch_size)
    # Remove dir if it already exists 
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    return save_dir

def get_fe_patch_dir(patch_size):
    dir=f'tmp/FE/'
    if patch_size[0]==1:
        dir+='2D/'
    else:
        dir+='3D/'
    dir+='batches/'
    return dir

def load_features(feature_dir):
    print('Loading features...')
    features = []

    files=[]
    for file in os.listdir(feature_dir):
        if file.endswith('.csv'):
            files.append(f'{feature_dir}{file}')
    files.sort()

    for file in files:     
        f=open(file, 'r')
        feature = f.read().strip('\n').split(',')
        features.append(list(feature))
        f.close()
    # Convert to numpy
    features = np.asarray(features, dtype=np.float64)

    return features

def save_partition(partition, save_dir):
    file = open(f'{save_dir}partition.pkl', 'wb')
    pickle.dump(partition, file)
    file.close()

def load_partition(prepare_batches, patch_size):
    load_dir=get_cae_patch_dir(prepare_batches, patch_size)
    file = open(f'{load_dir}partition.pkl', 'rb')
    partition = pickle.load(file)
    return partition

def get_cae_patch_dir(prepare_batches, patch_size):
    dir=f'tmp/CAE/'
    if 1 in patch_size:
        dir+='2D/'
    else:
        dir+='3D/'
    if prepare_batches:
        dir+='batches/'
    else:
        dir+='patches/'
    return dir


def ffr_values_to_target_list(ffr_dir, ffr_filename, pc_input_dir, ffr_cut_off=0.85):
    # Get all patients to classify
    classification_patients = get_sub_dirs(pc_input_dir)
    classification_patients = [x.split('_')[-1] for x in classification_patients]

    patient_values = defaultdict(list)
    
    # Read ffr lines from file
    with open(f'{ffr_dir}{ffr_filename}') as f:
        lines = f.readlines()

    for line in lines:
        line.rstrip('\n')
        split = line.split(' ')
        patient = split[0]

        # Add ffr value to dict if the patient is in the classification folder
        if patient in classification_patients:
            ffr = float(split[-1])
            patient_values[patient].append(ffr)
    target = []
    num = 0
    for values in patient_values.values():
        min_ffr = min(values)
        if min_ffr > ffr_cut_off:
            target.append(0)
            num += 1
        else:
            target.append(1)

    print('\nPatients with signficant stenosis: ', len(target)-num)
    print('Patients without signficant stenosis: ', num)

    return np.asarray(target)


def delete_tmp_files():
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')


def organize_data(data_dir):
    # Get all filenames
    filenames = get_nifti_filenames(data_dir, format=True)
    folders = []
    
    for filename in filenames:
        pattern = filename.split('Segmentation')[0]
        folders.append(pattern)
    count_dict = Counter(folders)
    
    for key in count_dict.keys():
        os.makedirs(f'{data_dir}{key[:-1]}/')
        names = list(filter(lambda x: x.startswith(key), filenames))
        for name in names:
            if name.endswith('LV.nii.gz'):
                shutil.move(f'{data_dir}{name}', f'{data_dir}{key[:-1]}/segmentation.nii.gz')
            elif name.endswith('CCTA.nii.gz'):
                shutil.move(f'{data_dir}{name}', f'{data_dir}{key[:-1]}/imaging.nii.gz' )


def create_sample_list(input_dir):
    # Get all patients to classify
    classification_patients = get_sub_dirs(input_dir)
    sample_list = {'TRAINING': classification_patients, 'VALIDATION': []}
    print(sample_list)
    # Save to json file
    with open(f'{input_dir}sample_list.json', 'w') as fp:
        json.dump(sample_list, fp)
    
def print_shapes(input_dir):
    img = get_paths_from_tree(input_dir, 'imaging')
    #clus = get_sub_dirs(input_dir, 'cluster')
    seg = get_paths_from_tree(input_dir, 'segmentation')

    for i in range(len(img)):
        print(f'\n{img[i]}')
        print('img_shape: ', nib.load(img[i]).get_data().shape)
        print('clus_shape: ', nib.load(seg[i]).get_data().shape)
    
def update_affine(img_filename, labelmap_filename):
    img = nib.load(img_filename)
    labelmap = nib.load(labelmap_filename)
    labelmap_data = labelmap.get_data()
    new_img = nib.Nifti1Image(labelmap_data, img.affine, img.header)
    nib.save(new_img, labelmap_filename)


def normalize_data(filenames, norm='z-score'):
    images = []
    headers = []
    affines = []
    norm_filenames = []
    for i in range(len(filenames)):
        img=nib.load(filenames[i])
        img_data=img.get_fdata()
        images.append(img_data)
        headers.append(img.header)
        affines.append(img.affine)
        norm_filenames.append(f'{filenames[i].rsplit("/", 1)[0]}/{norm}_norm.nii.gz')
    
    img_array=np.concatenate([x.ravel() for x in images])
    std=np.std(img_array)
    mean=np.mean(img_array)
    for i in range(len(images)):
        if norm=='minmax': n_img = (images[i] - img_array.min()) / (img_array.max() - img_array.min())
        if norm=='z-score': n_img= (images[i] - mean) / (std)
        new_img = nib.Nifti1Image(n_img, affines[i], headers[i])
        nib.save(new_img, norm_filenames[i])
    return norm_filenames

def resample_volume(img, voxel_size):
    resampled_img = nib.processing.resample_to_output(img, voxel_size)
    return resampled_img

def clipping(filenames, clipping=()):
    images = []
    headers = []
    affines = []
    norm_filenames = []
    for i in range(len(filenames)):
        img=nib.load(filenames[i])
        img_data=img.get_fdata()
        img_data = np.clip(img_data, int(clipping[0]), int(clipping[1]))    
        images.append(img_data)
        headers.append(img.header)
        affines.append(img.affine)
        norm_filenames.append(f'{filenames[i].rsplit("/", 1)[0]}/clipped_{str(clipping[0])}_{str(clipping[1])}.nii.gz')
    for i in range(len(images)):
        new_img = nib.Nifti1Image(images[i], affines[i], headers[i])
        nib.save(new_img, norm_filenames[i])
    return norm_filenames