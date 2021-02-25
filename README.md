# CAE Feature Extraction

## Program for extracting features from medical images

- Convolutional autoencoder (2D/3D)
- Automatic segmentation (from trained model) using miscnn
- Kmeans clustering
- Feature extraction clustered images 
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
       - [20181206\_ffr\_vals](classification/ffr_data/20181206_ffr_vals)
     - __patients__
       - __CT\_FFR\_25__
         - [imaging.nii.gz](classification/patients/CT_FFR_25/imaging.nii.gz)           (input)
         - [segmentation.nii.gz](classification/patients/CT_FFR_25/segmentation.nii.gz)   (output)
         - [cluster.nii.gz](classification/patients/CT_FFR_25/cluster.nii.gz)             (output)





     
The results from the automatic segmentaton and clustering will then be saved in the patients folder for classification (marked output)

### Automatic segmentation 
