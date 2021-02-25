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

├── data
    ├── CAE
    │   ├── CT_FFR_1
    │   │   ├── imaging.nii.gz          (input)
    │   │   └── segmentation.nii.gz     (output)
    └── classification
        ├── as_model
        │   └── model.best.hdf5
        ├── ffr_data
        │   └── 20181206_ffr_vals
        └── patients
            └── CT_FFR_2
                ├── imaging.nii.gz      (input)
                ├── segmentation.nii.gz (output)
                └── cluster.nii.gz      (output)


     
The results from the automatic segmentaton and clustering will then be saved in the patients folder for classification (marked output)

### Automatic segmentation 
