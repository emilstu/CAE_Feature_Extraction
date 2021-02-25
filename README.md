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
.
└── data
        ├── CAE
        │   └── patients
        │       └── case_00001
        │           └── imaging.nii.gz      (input)
        │           └── segmentation.nii.gz (input)     
        └── classification
            └── patients
            │   └── case_00002
            │       └── imaging.nii.gz      (input)
            │       └── segmentation.nii.gz (output)
            │       └── cluster.nii.gz      (output)
            └── as_model
            └── ffr_data
     
The results from the automatic segmentaton and clustering will then be saved in the patients folder for classification (marked output)

### Automatic segmentation 
