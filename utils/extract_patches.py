import numpy as np
import nibabel as nib
import torch
import torchio as tio
from sklearn.utils import check_array, check_random_state
import numbers
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm


def extract_2d_patches(img_filenames, patch_size, patch_overlap, min_labeled_pixels=None, labelmap_filenames=None):
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
    min_labeled_pixels : float between 0 and 1
        The minimum percentage of labeled pixels for a patch. If set to None patches are extracted based on center_voxel.
    labelmap_filenames : list of strings 
        Paths to labelmap
        
    Returns
    -------
    img_patches, label_patches : array, shape = (n_patches, patch_x, patch_y, patch_z, 1)
         The collection of patches extracted from the volumes, where `n_patches`
         is the total number of patches extracted.
    Examples
    --------
    TBD
    """
    
    print(f'\n\nExtracting patches from: {img_filenames}\n')
    img_patches = []
    label_patches = []
    for i in tqdm(range(len(img_filenames)), leave=False):
        if labelmap_filenames is not None:
            subject = tio.Subject(
                img=tio.Image(img_filenames[i], type=tio.INTENSITY),
                labelmap=tio.LabelMap(labelmap_filenames[i])
            )
        else:
            subject = tio.Subject(
                img=tio.Image(img_filenames[i], type=tio.INTENSITY)
            )

        sampler = tio.data.GridSampler(
            subject,
            patch_size,
            patch_overlap,
        )
        
        for patch in sampler:    
            # Extract patches based on labelmap
            if labelmap_filenames is not None:
                img_patch = np.array(patch.img.data)
                label_patch = np.array(patch.labelmap.data)
                # Extract patches based on labeled voxels
                if min_labeled_pixels is not None:
                    if torch.count_nonzero(patch.labelmap.data) >= patch_size[1] * patch_size[2] * min_labeled_pixels:
                        img_patches.append(img_patch)
                        label_patches.append(label_patch)
                # Extract patches based on min center voxel
                elif label_patch[0,0,int(patch_size[1]/2), int(patch_size[2]/2)] != 0:
                    img_patches.append(img_patch)
                    label_patches.append(label_patch)
            
            # Extract patches from whole image
            else:
                img_patch = np.array(patch.img.data)
                img_patches.append(img_patch)
    print(f'\tFinished extracting patches.')
    
    img_patches = np.array(img_patches).reshape(len(img_patches), patch_size[1], patch_size[2], 1)
    label_patches = np.array(label_patches).reshape(len(label_patches), patch_size[1], patch_size[2], 1)
    
    return img_patches, label_patches
    

#https://github.com/konopczynski/Vessel3DDL/blob/master/scripts/utils/patches_3d.py

def extract_3d_patches(img_filenames, patch_size, max_patches=None, random_state=None, labelmap_filenames=None, extract_labelmap_patches=False):
    """Reshape a 3D volumes into a collection of patches
    The resulting patches are allocated in a dedicated array.

    Parameters
    ----------
    img_filenames : list of strings  
        Paths to images to extract patches from 
    patch_size : tuple of ints (patch_x, patch_y, patch_z)
        the dimensions of one patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.
    extract_labelmap_patches : boolean
        Set to True if labelmap patches should be extracted 

    Returns
    -------
    img_all_patches, clus_all_patches : array, shape = (n_patches, patch_x, patch_y, patch_z, 1)
         The collection of patches extracted from the volume, where `n_patches`
         is either `len(img_filenames)*max_patches` or the total number of patches that can be
         extracted.
    """
    img_all_patches = []
    clus_all_patches = []

    print(f'\n\nExtracting patches from: {img_filenames}\n')
    for i in tqdm(range(len(img_filenames)), leave=False):
        img = nib.load(img_filenames[i])
        labelmap = nib.load(labelmap_filenames[i])
        
        volume = img.get_data()
        mask = labelmap.get_data()

        if extract_labelmap_patches:
            clus = labelmap.get_data()
        
        # Start patch extraction 
        v_x, v_y, v_z = volume.shape[:3]
        p_x, p_y, p_z = patch_size

        if p_x > v_x:
            raise ValueError("Height of the patch should be less than the height"
                            " of the volume.")

        if p_y > v_y:
            raise ValueError("Width of the patch should be less than the width"
                            " of the volume.")

        if p_z > v_z:
            raise ValueError("z of the patch should be less than the z"
                            " of the volume.")
                            
        volume = check_array(volume, allow_nd=True)
        volume = volume.reshape((v_x, v_y, v_z, -1))
        
        if extract_labelmap_patches:
            clus = check_array(clus, allow_nd=True)
            clus = clus.reshape((v_x, v_y, v_z, -1))
        
        n_colors = volume.shape[-1]
       
        # Extract patches 
        img_extracted_patches = extract_patches(volume, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)
        if extract_labelmap_patches:
            clus_extracted_patches = extract_patches(clus, patch_shape=(p_x, p_y, p_z, n_colors), extraction_step=1)

        n_patches = _compute_n_patches_3d(v_x, v_y, v_z, p_x, p_y, p_z, max_patches)
        # check the indexes where mask is True
        M=np.array(np.where(mask[int(p_x/2):int(v_x-p_x/2),
                                int(p_y/2):int(v_y-p_y/2),
                                int(p_z/2):int(v_z-p_z/2)]!=0)).T
        if max_patches:
            rng = check_random_state(random_state)
            indx = rng.randint(len(M), size=n_patches)
            i_s = M[indx][:,0]
            j_s = M[indx][:,1]
            k_s = M[indx][:,2]        
            img_patches = img_extracted_patches[i_s, j_s, k_s, 0]
            if extract_labelmap_patches:
                clus_patches = clus_extracted_patches[i_s, j_s, k_s, 0]
        else:
            img_patches = img_extracted_patches
            if extract_labelmap_patches:
                clus_patches = clus_extracted_patches

        # Reshape patches
        img_patches = img_patches.reshape(-1, p_x, p_y, p_z, n_colors)
        if extract_labelmap_patches:
            clus_patches = clus_patches.reshape(-1, p_x, p_y, p_z, n_colors)
        
        # Append patches to lis 
        img_all_patches = append_patches_to_list(img_patches, img_all_patches)
        if extract_labelmap_patches:
            clus_all_patches = append_patches_to_list(clus_patches, clus_all_patches)
    
    print(f'\tFinished extracting patches.')
    
    img_all_patches = np.array(img_all_patches)
    clus_all_patches = np.array(clus_all_patches)

    print(clus_all_patches.shape)


    return img_all_patches, clus_all_patches



def _compute_n_patches_3d(i_x, i_y, i_z, p_x, p_y, p_z, max_patches=None):
    """Compute the number of patches that will be extracted in a volume.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    i_x : int
        The number of voxels in x dimension
    i_y : int
        The number of voxels in y dimension
    i_z : int
        The number of voxels in z dimension
    p_x : int
        The number of voxels in x dimension of a patch
    p_y : int
        The number of voxels in y dimension of a patch
    p_z : int
        The number of voxels in z dimension of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """

    n_x = i_x - p_x + 1
    n_y = i_y - p_y + 1
    n_z = i_z - p_z + 1
    all_patches = n_x * n_y * n_z

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches

def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted
    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.
    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches
    

def append_patches_to_list(patches, append_list):
    res = append_list
    for i in range(patches.shape[0]):
        res.append(patches[i])
    return res



