# CAE_Feature_Extraction
A program for feature extraction and classification of LV-myocardium properties in relation to measured FFR (Fractional Flow Reserve). The program consist of 

- Convolutional autoencoder (2D/3D) for dimensionality reduction of image patches
- Automatic segmentation (from a trained model) using [MIScnn](https://github.com/frankkramer-lab/MIScnn)
- Kmeans clustering
- Feature extraction
- SVM classification of extracted features 

## Folder structure
Recomended data structure
- __data__
   - __CAE__
     - __patients__
       - __CT\_FFR\_10__
         - [imaging.nii.gz](CAE/patients/CT_FFR_10/imaging.nii.gz) (input)
         - [segmentation.nii.gz](CAE/patients/CT_FFR_10/segmentation.nii.gz)(input)            
   - __classification__
     - __as\_model__
       - [model.best.hdf5](classification/as_model/model.best.hdf5)
     - __ffr\_data__
       - [ffr\_vals](classification/ffr_data/ffr_vals)
     - __patients__
       - __CT\_FFR\_25__
         - [imaging.nii.gz](classification/patients/CT_FFR_25/imaging.nii.gz)(input)
         - [segmentation.nii.gz](classification/patients/CT_FFR_25/segmentation.nii.gz)(output)
         - [cluster.nii.gz](classification/patients/CT_FFR_25/cluster.nii.gz)(output)

The results from the automatic segmentation and clustering will be saved in the patient folder for classification (marked output)
## Convolutional Autoencoder 
### 2D-CAE
To utilize 2D autoencoder the patch-size has to be on the form 
```bash
cae_patch_size = (1, 48, 48)
```
A patch-overlap has to be specified
```bash
patch_overlap = (0, 40, 40)
```
Additionally, the minimum number of labeled voxels for each patch has to be specified
```bash
min_labeled_pixels = 0.5
```
This indicates that at least 50 % of the voxels from a patch have to be labeled as segmentation for the CAE to use it for training/predicting. 
### 3D-CAE
To utilize 3D autoencoder the patch-size has to be on the form 
```bash
cae_patch_size = (160, 160, 160)
```
A maximum number of patches can be specified
```bash
max_patches = 100000
```
otherwise, patches will be extracted with an overlap of (x-1,y-1,z-1). Also for the 3D autoencoder, the minimum number of labeled voxels has to be specified.

## Patient Classification
For the programs working on the classification data, the input directory has to be specified
```bash
pc_input_dir = 'data/classification/patients/'
```
### Automatic segmentation
The automatic segmentation is based on [MIScnn](https://github.com/frankkramer-lab/MIScnn), and is utilized on the classification data.

Directory paths example:
```bash
as_model_name = 'model.best'
as_model_dir = 'data/classification/as_model/'
```

### Clustering
k-means clustering of segmentations to be used for the feature extraction. Two parameters must be specified
```bash
num_iters = 100
num_clusters = 500
```
which is the number of iterations in the main loop of k-means, and the number of clusters the segmentaions should be clustered into.

### Feature Extraction
Features are extracted from the clusters by utilizing a trained CAE model (2D/3D). For each cluster, the maximum standard deviation is calculated. The result is a 1D list with the same size as the number of clusters. One parameter must be specified, which can take two possible values 
```bash
voxel_selection = 'center'
```
which selects a cluster for a specific patch based on the center index of the patch, or
```bash
voxel_selection = 'highest_share'
```
which selects a cluster for a specific patch based on the highest share of voxels. If the background has the highest share of voxels, the patch isn't used. 

Directory paths example:
```bash
fe_model_dir = 'evaluation/CAE/2D/ex2/'
fe_model_name = 'model_2D'
```
### SVM-classification 
The extracted features are classified using Support Vector Machines. Patients are labeled based on ffr measurements according to a specified cut-of-value
```bash
ffr_boundary = 0.85
```
Additionally, the name of the ffr-file must be specified 
```bash
ffr_filename = 'ffr_vals'
```
A sequence of lines in the ffr_file can be
```bash
...
20 11 0.89
20 13 0.90
31 2 0.77
32 7 0.91
...
```
The first column is the patient number, and the last is the ffr-values. The second column is ignored. If more than one ffr-value exists for a patient the smallest value is chosen. 

Directory paths example:
```bash
feature_dir = 'evaluation/classification/features/ex1/'
ffr_dir = 'data/classification/ffr_data/'
```
## Running the program
To start the program execute the main script
```bash
python3 main.py
```


