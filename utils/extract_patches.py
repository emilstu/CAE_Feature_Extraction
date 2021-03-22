import numpy as np
import nibabel as nib
import torch
import torchio as tio
import numbers
from tqdm import tqdm
from utils import util
import os 
import shutil
import math


def patch_sampler(img_filenames, labelmap_filenames, patch_size, sampler_type, out_dir, max_patches=None, voxel_spacing=(), patch_overlap=(0,0,0), min_labeled_voxels=1.0, label_prob=0.8, save_patches=False, batch_size=None, prepare_batches=False, inference=False):
    """Reshape a 3D volumes into a collection of 2D patches
    The resulting patches are allocated in a dedicated array.
    
    Parameters
    ----------
    img_filenames : list of strings  
        Paths to images to extract patches from 
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        The dimensions of one patch
    patch_overlap : tuple of ints (0, patch_x, patch_y)
        The maximum patch overlap between the patches 
    min_labeled_voxels is not None: : float between 0 and 1
        The minimum percentage of labeled pixels for a patch. If set to None patches are extracted based on center_voxel.
    labelmap_filenames : list of strings 
        Paths to labelmap
        
    Returns
    -------
    img_patches, label_patches : array, shape = (n_patches, patch_x, patch_y, patch_z, 1)
         The collection of patches extracted from the volumes, where `n_patches`
         is the total number of patches extracted.
    """

    if max_patches is not None: max_patches = int(max_patches/len(img_filenames))    
    img_patches = []
    label_patches = []
    patch_counter = 0
    save_counter = 0
    img_ids = []
    label_ids = []
    save_size = 1
    if prepare_batches: save_size = batch_size
    print(f'\nExtracting patches from: {img_filenames}\n')
    for i in tqdm(range(len(img_filenames)), leave=False):
        if voxel_spacing: util.update_affine(img_filenames[i], labelmap_filenames[i])
        if labelmap_filenames:
            subject = tio.Subject(img=tio.Image(img_filenames[i], type=tio.INTENSITY), labelmap=tio.LabelMap(labelmap_filenames[i]))
        # Apply transformations
        #transform = tio.ZNormalization()
        #transformed = transform(subject)
        transform = tio.RescaleIntensity((0, 1))
        transformed = transform(subject)
        if voxel_spacing:
            transform = tio.Resample(voxel_spacing)
            transformed = transform(transformed)
        num_img_patches = 0
        if sampler_type == 'grid':
            sampler = tio.data.GridSampler(transformed, patch_size, patch_overlap)
            for patch in sampler:
                img_patch = np.array(patch.img.data)
                label_patch = np.array(patch.labelmap.data)
                labeled_voxels = torch.count_nonzero(patch.labelmap.data) >= patch_size[0]*patch_size[1]*patch_size[2]*min_labeled_voxels
                center = label_patch[0, int(patch_size[0]/2),int(patch_size[1]/2), int(patch_size[2]/2)] != 0
                if labeled_voxels or center:
                    img_patches.append(img_patch)
                    label_patches.append(label_patch)
                    patch_counter += 1
                    num_img_patches += 1
                if save_patches: img_patches, label_patches, img_ids, label_ids, save_counter, patch_counter = save(    img_patches, label_patches, 
                                                                                                                        img_ids, label_ids,
                                                                                                                        save_counter, patch_counter, 
                                                                                                                        save_size, patch_size,
                                                                                                                        inference, out_dir  )                                                                    
                # Check if max_patches for img
                if max_patches is not None:
                    if num_img_patches > max_patches:
                        break
        else:
            # Define sampler
            one_label=1.0-label_prob
            label_probabilities = {0: one_label, 1: label_prob}
            sampler = tio.data.LabelSampler(patch_size, label_probabilities=label_probabilities)
            if max_patches is None:
                generator = sampler(transformed)
            else:
                generator = sampler(transformed, max_patches)
            for patch in generator:
                img_patches.append(np.array(patch.img.data))
                label_patches.append(np.array(patch.labelmap.data))
                patch_counter += 1
                if save_patches: img_patches, label_patches, img_ids, label_ids, save_counter, patch_counter = save(    img_patches, label_patches, 
                                                                                                                        img_ids, label_ids,
                                                                                                                        save_counter, patch_counter, 
                                                                                                                        save_size, patch_size,
                                                                                                                        inference, out_dir  )
    print(f'Finished extracting patches.')
    if save_patches:
        return img_ids, label_ids
    else:
        if patch_size[0] == 1:
            return np.array(img_patches).reshape(len(img_patches), patch_size[1], patch_size[2], 1), np.array(label_patches).reshape(len(label_patches), patch_size[1], patch_size[2], 1)
        else:
            return np.array(img_patches).reshape(len(img_patches), patch_size[0], patch_size[1], patch_size[2], 1), np.array(label_patches).reshape(len(label_patches), patch_size[1], patch_size[2], 1)


def save(img_patches, label_patches, img_ids, label_ids, save_counter, patch_counter, save_size, patch_size, inference, out_dir):
    if save_size != patch_counter:
        return img_patches, label_patches, img_ids, label_ids, save_counter, patch_counter
    else:
        img_patches = np.array(img_patches)
        label_patches = np.array(label_patches)
        if save_size == 1:
            if patch_size[0]==1:
                img_patches = img_patches.reshape(patch_size[1], patch_size[2], 1)
                label_patches = label_patches.reshape(patch_size[1], patch_size[2], 1)
            else:
                img_patches = img_patches.reshape(patch_size[0], patch_size[1], patch_size[2], 1)
                label_patches = label_patches.reshape(patch_size[0], patch_size[1], patch_size[2], 1)
            np.save(f'{out_dir}img_patch_{save_counter}.npy', img_patches.astype('float32'))
            img_ids.append(f'img_patch_{save_counter}')
            if inference:
                np.save(f'{out_dir}clus_patch_{save_counter}.npy', label_patches.astype('float32'))
                label_ids.append(f'clus_patch_{save_counter}')
        else:
            if patch_size[0]==1:
                img_patches = img_patches.reshape(len(img_patches), patch_size[1], patch_size[2], 1)
                label_patches = label_patches.reshape(len(label_patches), patch_size[1], patch_size[2], 1)
            else:
                img_patches = img_patches.reshape(len(img_patches), patch_size[0], patch_size[1], patch_size[2], 1)
                label_patches = label_patches.reshape(len(label_patches), patch_size[0], patch_size[1], patch_size[2], 1)
            np.save(f'{out_dir}/img_batch_{save_counter}.npy', img_patches.astype('float32'))
            img_ids.append(f'img_batch_{save_counter}')
            if inference:
                np.save(f'{out_dir}clus_batch_{save_counter}.npy', label_patches.astype('float32'))
                label_ids.append(f'clus_batch_{save_counter}')
        save_counter += 1
        patch_counter = 0
        
        return [], [], img_ids, label_ids, save_counter, patch_counter