# CAE Feature Extraction

## Program for extracting features from medical images

- Convolutional autoencoder (2D/3D)
- Automatic segmentation (from trained model) using miscnn
- Kmeans clustering
- Feature extraction on clustered images 
- SVM classification of extracted features 

## Getting started 

### Folder structure
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


The results from the automatic segmentaton and clustering will be saved in the patients folder for classification (marked output)

### CAE
Convolutional autoencoder for dimensionality reduction of image patches. The CAE can be trained using either 2D and 3D patches. Utilized on CAE data, where manual segmentations are available.

### Automatic segmentation
The automatic segmentation is based on [miscnn](https://github.com/frankkramer-lab/MIScnn), and is utilized on the classification data.

### Clustering (k-means)
Clustering of segmentations to be used for the feature extraction. 

### Feature Extraction
Features are extracted from the clusters by utilizing a trained CAE model (2D/3D). For each cluster the maximum standard deviation is calculated. The result is a 1D list with the same size as number of clusters.

### SVM-classification 
The extracted features are loaded and  are classified by Support Vector Machines. Patients are labeled based on ffr measurements according to a specified cut-of-value.

## Setting parameters
F
### 2D-CAE
To use utilize 2D autoencoder the patch-size has to be on the form 
```bash
cae_patch_size = (1, 48, 48)
```
and a patch-overlap has to be specified
```bash
patch_overlap=(0, 40, 40)
```
for a patch size of 48x48 and maximum overlap of 40 in both directions. Additionally the minimum number of labeled voxels for each patch has to be specified
```bash
min_labeled_pixels=0.5
```
which indicates that at least 50 % of the voxels from a patch has to be labeled as segmentation for the CAE to use it for training/predicting. 

### 3D-CAE
To use utilize 3D autoencoder the patch-size has to be on the form 
```bash
cae_patch_size = (160, 160, 160)
```
A maximum number of patches can be specified
```bash
max_patches=100000
```
otherwise, patches will be extracted with a overlap of (x-1,y-1,z-1). Also for the 3D autoencoder the minimum number of labeled voxels has to be specified.

### Clustering



## Running the program
To start the program execute the main script
```bash
python3 main.py
```


