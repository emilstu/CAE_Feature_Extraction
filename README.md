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
├── CAE
│   ├── __init__.py
│   ├── cae.py
│   ├── cae_2d.py
│   └── cae_3d.py
├── automatic_segmentation.py
├── clustering.py
├── data
│   ├── CAE
│   │   ├── CT_FFR_10
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_11
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_12
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_13
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_14
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_15
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_16
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_17
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_18
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_19
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_2
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_20
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_21
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_22
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_23
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_24
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_3
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_4
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_5
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_6
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_65
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_66
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_67
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_68
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_69
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_7
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   ├── CT_FFR_8
│   │   │   ├── imaging.nii.gz
│   │   │   └── segmentation.nii.gz
│   │   └── CT_FFR_9
│   │       ├── imaging.nii.gz
│   │       └── segmentation.nii.gz
│   └── classification
│       ├── as_model
│       │   └── model.best.hdf5
│       ├── ffr_data
│       │   ├── 20181206_ffr_vals
│       │   └── ffr_values.txt
│       └── patients
│           ├── CT_FFR_25
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_26
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_27
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_28
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_29
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_31
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_32
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_33
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_34
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_35
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_36
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_37
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_38
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_39
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_40
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_42
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_43
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_44
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_45
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_46
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_47
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_48
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_49
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_50
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_51
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_52
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_53
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_54
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_55
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_56
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_57
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_58
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_59
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_60
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_61
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_62
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_63
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_64
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_65
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_66
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_67
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_68
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           ├── CT_FFR_69
│           │   ├── imaging.nii.gz
│           │   └── segmentation.nii.gz
│           └── sample_list.json
├── evaluation
│   ├── CAE
│   │   ├── 2D
│   │   │   ├── ex1
│   │   │   │   ├── model_2D.h5
│   │   │   │   ├── model_2D.json
│   │   │   │   ├── org.png
│   │   │   │   ├── rec.png
│   │   │   │   ├── test_samples
│   │   │   │   └── train_samples
│   │   │   ├── ex2
│   │   │   │   ├── fitting_curve.png
│   │   │   │   ├── model_2D.h5
│   │   │   │   ├── model_2D.json
│   │   │   │   ├── org.png
│   │   │   │   ├── rec.png
│   │   │   │   ├── test_samples
│   │   │   │   └── train_samples
│   │   │   ├── ex3
│   │   │   │   ├── fitting_curve.png
│   │   │   │   ├── model_tex.h5
│   │   │   │   ├── model_tex.json
│   │   │   │   ├── org.png
│   │   │   │   ├── rec.png
│   │   │   │   ├── test_samples
│   │   │   │   └── train_samples
│   │   │   ├── ex4
│   │   │   │   ├── fitting_curve.png
│   │   │   │   ├── model_2D.h5
│   │   │   │   ├── model_2D.json
│   │   │   │   ├── org.png
│   │   │   │   ├── rec.png
│   │   │   │   ├── test_samples
│   │   │   │   └── train_samples
│   │   │   ├── ex5
│   │   │   │   ├── fitting_curve.png
│   │   │   │   ├── model_2D.h5
│   │   │   │   ├── model_2D.json
│   │   │   │   ├── org.png
│   │   │   │   ├── rec.png
│   │   │   │   ├── test_samples
│   │   │   │   └── train_samples
│   │   │   └── ex6
│   │   │       ├── fitting_curve.png
│   │   │       ├── model_2D.h5
│   │   │       ├── model_2D.json
│   │   │       ├── org.png
│   │   │       ├── rec.png
│   │   │       ├── test_samples
│   │   │       └── train_samples
│   │   └── 3D
│   │       ├── ex1
│   │       ├── ex2
│   │       ├── ex3
│   │       ├── ex4
│   │       └── ex5
│   └── classification
│       ├── features
│       └── svm
├── feature_extraction.py
├── main.py
├── requirements.txt
├── svm_classifier.py
├── tree.txt
└── utils
    ├── __init__.py
    ├── extract_patches.py
    ├── kmeans.py
    └── util.py

97 directories, 202 files


     
The results from the automatic segmentaton and clustering will then be saved in the patients folder for classification (marked output)

### Automatic segmentation 
