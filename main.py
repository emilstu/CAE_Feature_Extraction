from numpy.lib.function_base import delete
from automatic_segmentation import AutomaticSegmentation
from clustering import Clustering
from feature_extraction import FeatureExtraction
from svm_classifier import SvmClassifier
from CAE.cae import CAE
from utils import util

#-----------------------------------------------------#
#               Program Parameters                    #
#-----------------------------------------------------#
autoencoder = True
automatic_segmentation = True
cluster = True
extract_features = True
classify = True
delete_tmp = True

#-----------------------------------------------------#
#             2D/3D Convolutional Autoencoder         #
#-----------------------------------------------------#
cae_input_dir = 'data/CAE/'
cae_model_dir = ''

#cae_patch_size = (200, 200, 200)
cae_patch_size = (1, 48, 48) 
patch_overlap=(0, 0, 0)
min_labeled_pixels=0.7
max_patches = 20

batch_size = 500
epochs = 1
batches_per_epoch = 1
cae_test_size = 0.5

#-----------------------------------------------------#
#               Patient Classification                #
#-----------------------------------------------------#
pc_input_dir = 'data/classification/patients/'
pc_test_size = 0.2

# Automatic Segmentstion
as_model_name = 'model.best'
as_model_dir = 'data/classification/as_model/'
as_patch_size = (160, 160, 80) 

# Clustering
num_iters = 5
num_clusters = 4

# Feature Extraction
fe_model_dir = 'evaluation/CAE/2D/ex2/'
fe_model_name = 'model_2D'
voxel_selection = 'center' #'highest_share'

# SVM Classification
feature_dir = 'evaluation/classification/features/ex1/'
ffr_dir = 'data/classification/ffr_data/'
ffr_filename = '20181206_ffr_vals'
ffr_boundary = 0.85

#-----------------------------------------------------#
#             2D/3D Convolutional Autoencoder         #
#-----------------------------------------------------#
if autoencoder:
    cae = CAE(  patch_size=cae_patch_size, 
                patch_overlap=patch_overlap,
                min_labeled_pixels=min_labeled_pixels,
                test_size=cae_test_size,
                input_dir=cae_input_dir,
                model_dir=cae_model_dir )

    cae.train(batch_size, epochs, batches_per_epoch)
    cae.predict(batch_size, delete_patches=True)


#-----------------------------------------------------#
#               Patient classification                #
#-----------------------------------------------------#
if automatic_segmentation:
    asg = AutomaticSegmentation(    model_name=as_model_name,
                                    patch_size=as_patch_size,
                                    input_dir=pc_input_dir, 
                                    model_dir=as_model_dir   )
    
    asg.run()
    asg.run_postprocessing()


if cluster:
    clustering = Clustering(    num_iters=num_iters,
                                num_clusters=num_clusters,
                                input_dir=pc_input_dir  )
    
    clustering.run() 


if extract_features:
    fe = FeatureExtraction( model_name=fe_model_name,
                            patch_size=cae_patch_size,
                            patch_overlap=patch_overlap,
                            min_labeled_pixels=min_labeled_pixels,
                            num_clusters=num_clusters,
                            voxel_selection=voxel_selection,
                            max_patches=max_patches,
                            model_dir=fe_model_dir,
                            input_dir=pc_input_dir  )
    
    fe.run(batch_size=batch_size)


if classify:
    svm = SvmClassifier(    feature_dir=feature_dir,
                            ffr_dir=ffr_dir,
                            ffr_filename=ffr_filename,
                            input_dir=pc_input_dir,
                            ffr_boundary=ffr_boundary,   
                            test_size=pc_test_size  )
                            
    svm.train()
    svm.predict()

if delete_tmp:
    util.delete_tmp_files()
    

