import numpy as np
import glob
import nibabel as nib
from keras.models import model_from_json 
import pickle
import os
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
import shutil
from collections import Counter
from collections import defaultdict
import json

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


def save_features(features, patient, out_dir): 
    # Add results to dict
    fe_dict = {patient: features}
    
    # Append feature to file
    with open(f'{out_dir}features.csv', 'a') as f:
        for key in fe_dict.keys():
            f.write("%s: %s\n"%(key,fe_dict[key]))
    
def save_dict(dict, out_dir, filename):
    with open(f'{out_dir}{filename}', 'w') as f:
        for key in dict.keys():
            f.write("%s: %s\n"%(key,dict[key]))

def load_features(feature_dir):
    print('Loading features...')
    features = []
    with open(f'{feature_dir}features.csv', 'r') as f:
        for row in f:
            feature = row.split(' ', 1)[1]
            feature = feature.strip(']\n[').split(', ')
            features.append(feature)
    print('Loaded features')
    # Convert to numpy 
    features = np.asarray(features, dtype=np.float64)
    return features

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
        
    print(patient_values)
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

    return target


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
    
