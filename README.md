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


The results from the automatic segmentaton and clustering will then be saved in the patients folder for classification (marked output)

### CAE
Convolutional autoencoder for dimensionality reduction of image patches. The CAE can be trained using either 2D and 3D patches. Utilized on CAE data, where manual segmentations are available.

### Automatic segmentation
The automatic segmentation is based on [miscnn](https://github.com/frankkramer-lab/MIScnn), and is utilized on the classification data.

### Clustering (k-means)
Clustering of segmentations to be used for the feature extraction. 

### Feature Extraction
Features are extracted from the clusters using a trained CAE model (2D/3D). For each cluster the maximum standard deviation is calculated. The result is a 1D list with the same size as number of clusters.  

### SVM-classification 
The extracted features are classified by Support Vector Machines. The patients are labled based on ffr measurements.



