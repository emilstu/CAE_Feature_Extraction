# CAE_Feature_Extraction
A program for feature extraction and classification of LV-myocardium properties in relation to measured FFR (Fractional Flow Reserve). The program consists of 

- Convolutional autoencoder (2D/3D) for dimensionality reduction of image patches
- Automatic segmentation (from a trained model) using [MIScnn](https://github.com/frankkramer-lab/MIScnn)
- Kmeans clustering
- Feature extraction
- Classification of extracted features 

## Getting started
The necessary packages and dependencies can be found in the requirements.yml file. It can be installed through a virtual conda environment
```bash
$ conda env create -f requirements.yml
```
### Recommended data structure
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

## Running the program
All the data directories are by default set according to the recommended data structure. For programs using results from other programs (FeEx and SVM) some directories have to be specified.
### 2D/3D Convolutional Autoencoder
```bash
$ python3 main.py CAE -h

usage: main.py CAE [-h] [-dd DATA_DIR] -ps PATCH_SIZE -po PATCH_OVERLAP -st {grid,label} [-mlv MIN_LAB_VOX] [-lb LABEL_PROB] [-mp MAX_PATCHES] [-r RESAMPLE] [-c CLIPPING]
                   [-pn {z-score,minmax,abs}] [-e EPOCHS] [-bs BATCH_SIZE] [-ts TEST_SIZE] [-pb] [-ld] [-md MODEL_DIR]

optional arguments:
  -h, --help                                                   show this help message and exit
  -dd DATA_DIR, --data_dir DATA_DIR                            Directory where data is stored.
  -ps PATCH_SIZE, --patch_size PATCH_SIZE                      Patch size 2D/3D: "(1,int,int)" or "(int,int,int)".
  -po PATCH_OVERLAP, --patch_overlap PATCH_OVERLAP             Patch overlap 2D/3D: (0,int,int) or (int,int,int). Must be even number and smaller than patch size.
  -st {grid,label}, --sampler_type {grid,label}                Sampler type
  -mlv MIN_LAB_VOX, --min_lab_vox MIN_LAB_VOX                  Minimum labled voxels used by grid-sampler.
  -lb LABEL_PROB, --label_prob LABEL_PROB                      Probability of choosing patches with labeled voxel as center. Used by label-sampler.
  -mp MAX_PATCHES, --max_patches MAX_PATCHES                   Maximum number of patches to extract.
  -r RESAMPLE, --resample RESAMPLE                             Resample to common voxel spacing "(float,float,float)".
  -c CLIPPING, --clipping CLIPPING                             Clipping range: "(int, int)"
  -pn {z-score,minmax,abs}, --pixel_norm {z-score,minmax,abs}  Pixel normalization type
  -e EPOCHS, --epochs EPOCHS                                   Number of epochs in training of CAE.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE                      Batch size for training.
  -ts TEST_SIZE, --test_size TEST_SIZE                         CAE test size. Float between 0.0 and 1.0.
  -pb, --prepare_batches                                       Specified when batches should be prepared and saved as mini-batches.
  -ld, --load_data                                             Specified when patches should be loaded.
  -md MODEL_DIR, --model_dir MODEL_DIR                         Directory where model is stored. When specified predictions are made on the loaded model.
```
### Automatic segmentation
The automatic segmentation is based on [MIScnn](https://github.com/frankkramer-lab/MIScnn). 
```bash
$ python3 main.py AutSeg -h

usage: main.py AutSeg [-h] [-dd DATA_DIR] [-md MODEL_DIR] -mn MODEL_NAME -ps PATCH_SIZE -po PATCH_OVERLAP

optional arguments:
  -h, --help                                        Show this help-message and exit.
  -dd DATA_DIR, --data_dir DATA_DIR                 Directory where data is stored.
  -md MODEL_DIR, --model_dir MODEL_DIR              Directory where model is stored.
  -mn MODEL_NAME, --model_name MODEL_NAME           Model name, i.e. "model.best".
  -ps PATCH_SIZE, --patch_size PATCH_SIZE           Patch size used when the model was trained: "(int,int,int)".
  -po PATCH_OVERLAP, --patch_overlap PATCH_OVERLAP  Patch overlap: "(int,int,int)"lap PATCH_OVERLAP  Patch overlap: "(int,int,int)".
```
### Clustering
k-means clustering of segmentations to be used for the feature extraction.
```bash
python3 main.py CLUS -h

usage: main.py CLUS [-h] [-dd DATA_DIR] -i ITERATIONS -nc NUM_CLUSTERS

optional arguments:
  -h, --help                                     Show this help-message and exit.
  -dd DATA_DIR, --data_dir DATA_DIR              Directory where data is stored.
  -i ITERATIONS, --iterations ITERATIONS         Number of iterations to run k-means clustering.
  -nc NUM_CLUSTERS, --num_clusters NUM_CLUSTERS  Number of clusters.
```
### Feature Extraction
Features are extracted from the clusters by utilizing a trained CAE model (2D/3D). For each cluster, the maximum standard deviation is calculated. The result is a 1D list with the same size as the number of clusters.
```bash
$ python3 main.py FeEx -h

usage: main.py FEX [-h] [-dd DATA_DIR] -md MODEL_DIR -mn MODEL_NAME -ps PATCH_SIZE [-po PATCH_OVERLAP] [-mlv MIN_LAB_VOX] [-cs {center,highest_share}] -nc NUM_CLUSTERS
                   [-r RESAMPLE] [-c CLIPPING] [-pn {z-score,minmax,abs}] -eln ENCODED_LAYER_NUM [-spn START_PATIENT_NUM] [-sp]

optional arguments:
  -h, --help                                                              show this help message and exit
  -dd DATA_DIR, --data_dir DATA_DIR                                       Directory where data is stored.
  -md MODEL_DIR, --model_dir MODEL_DIR                                    Directory where model is stored.
  -mn MODEL_NAME, --model_name MODEL_NAME                                 Model name, i.e. "model_2D".
  -ps PATCH_SIZE, --patch_size PATCH_SIZE                                 Patch size 3D/3D: "(1,int,int)" or "(int,int,int)".
  -po PATCH_OVERLAP, --patch_overlap PATCH_OVERLAP                        Patch overlap 2D/3D: "(0,int,int)" or "(int,int,int)". Must be even number and smaller than patch size.
  -mlv MIN_LAB_VOX, --min_lab_vox MIN_LAB_VOX                             Minimum labled voxels used by grid-sampler.
  -cs {center,highest_share}, --cluster_selection {center,highest_share}  Method used to select which cluster a specific patch belongs to.
  -nc NUM_CLUSTERS, --num_clusters NUM_CLUSTERS                           Number og clusters used in the images to extract features from.
  -r RESAMPLE, --resample RESAMPLE                                        Resample to common voxel spacing "(float,float,float)".
  -c CLIPPING, --clipping CLIPPING                                        Clipping range: "(int, int)"
  -pn {z-score,minmax,abs}, --pixel_norm {z-score,minmax,abs}             Pixel normalization type
  -eln ENCODED_LAYER_NUM, --encoded_layer_num ENCODED_LAYER_NUM           Number of the encoded layer in CAE-architecture counting from the bottom.
  -spn START_PATIENT_NUM, --start_patient_num START_PATIENT_NUM           Starting point of the patient in patient data dir.
  -sp, --save_patches                                                     Specified if patches should be saved to disk.
```
Center-selection selects a cluster for a specific patch based on the center index of the patch. Highest_share-selection selects a cluster for a specific patch based on the highest share of voxels. If the background has the highest share of voxels, the patch isn't used. 
### Classification 
The extracted features are classified using Support Vector Machines. Patients are labeled based on ffr measurements according to a specified cut-of-value
```bash
$ python3 main.py CLA -h

usage: main.py CLA [-h] [-dd DATA_DIR] -fd FEATURE_DIR [-ffd FFR_DIR] -ffn FFR_FILENAME [-ffc FFR_CUT_OFF] [-f FOLDS] [-i ITERATIONS] [-fs {train,entire}] [-c2 CHI2]
                   [-mi MUTUAL_INFORMATION] [-plt]

optional arguments:
  -h, --help                                                       show this help message and exit
  -dd DATA_DIR, --data_dir DATA_DIR                                Directory where data is stored.
  -fd FEATURE_DIR, --feature_dir FEATURE_DIR                       Directory where features are stored, i.e. output from FEX.
  -ffd FFR_DIR, --ffr_dir FFR_DIR                                  Directory ffr_values are stores.
  -ffn FFR_FILENAME, --ffr_filename FFR_FILENAME                   Filename for the file where ffr-values are stored.
  -ffc FFR_CUT_OFF, --ffr_cut_off FFR_CUT_OFF                      Filename for the file where ffr-values are stored.
  -f FOLDS, --folds FOLDS                                          Number of folds each iteration in cross-validation.
  -i ITERATIONS, --iterations ITERATIONS                           Number of iterations to run cross-validation.
  -fs {train,entire}, --feature_selection {train,entire}           Specification of FS should be performed on train set vs entire dataset
  -c2 CHI2, --chi2 CHI2                                            kbest chi2. Default setting is set to all
  -mi MUTUAL_INFORMATION, --mutual_information MUTUAL_INFORMATION  kbest mutual information. Default is set to all
  -plt, --plot                                                     Specified if feature statistics (chi2 and MI) should be plotted
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
The first column is the patient number, and the last is the ffr-values. The second column is ignored. If more than one ffr-value exists for a patient the smallest value is chosen. All the patients in the classification folder have to be represented in the file.


