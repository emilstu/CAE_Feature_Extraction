from numpy.lib.function_base import delete
from automatic_segmentation import AutomaticSegmentation
from clustering import Clustering
from feature_extraction import FeatureExtraction
from svm_classifier import SvmClassifier
from CAE.cae import CAE
from utils import util


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
test_size = 0.5

#-----------------------------------------------------#
#               Automatic Segmentation                #
#-----------------------------------------------------#
as_model_name = 'model.best'
as_model_dir = 'data/classification/as_model/'
as_train_input_dir = 'data/classification/training/'
as_pred_input_dir = 'data/classification/prediction/'
as_patch_size = (160, 160, 80) 

#-----------------------------------------------------#
#                   Clustering                        #
#-----------------------------------------------------#
clus_train_input_dir = 'data/classification/training/'
clus_pred_input_dir = 'data/classification/prediction/'

num_iters = 5
num_clusters = 4

#-----------------------------------------------------#
#               Feature Extraction                    #
#-----------------------------------------------------#
fe_train_input_dir = 'data/classification/training/'
fe_pred_input_dir  = 'data/classification/prediction/'
fe_model_dir = 'evaluation/CAE/2D/ex2/'

fe_model_name = 'model_2D'
voxel_selection = 'center' #'highest_share'

#-----------------------------------------------------#
#               SVM Classification                    #
#-----------------------------------------------------#
svm_train_feature_dir = 'evaluation/classification/features/training/ex1/'
svm_train_target_dir = 'data/classification/training/'
svm_pred_feature_dir = 'evaluation/classification/features/prediction/ex1/'
svm_pred_target_dir = 'data/classification/prediction/'
ffr_boundary = 0.85

#-----------------------------------------------------#
#               Program Parameters                    #
#-----------------------------------------------------#
autoencoder = False
automatic_segmentation = False
cluster = False
extract_features = False
classify = True
delete_tmp = True


if autoencoder:
    cae = CAE(  patch_size=cae_patch_size, 
                patch_overlap=patch_overlap,
                min_labeled_pixels=min_labeled_pixels,
                test_size=test_size,
                input_dir=cae_input_dir,
                model_dir=cae_model_dir )

    cae.train(batch_size, epochs, batches_per_epoch)
    cae.predict(batch_size, delete_patches=True)
    

if automatic_segmentation:
    asg = AutomaticSegmentation(    model_name=as_model_name,
                                    patch_size=as_patch_size,
                                    input_train_dir=as_train_input_dir,
                                    input_pred_dir=as_pred_input_dir, 
                                    model_dir=as_model_dir   )
    
    asg.run(type='training')
    asg.run_postprocessing()

    asg.run(type='prediction')
    asg.run_postprocessing()


if cluster:
    clustering = Clustering(    num_iters=num_iters,
                                num_clusters=num_clusters,
                                train_input_dir=clus_train_input_dir,
                                pred_input_dir=clus_pred_input_dir  )

    clustering.run(type='training')     
    clustering.run(type='prediction') 




if extract_features:
    fe = FeatureExtraction( model_name=fe_model_name,
                            patch_size=cae_patch_size,
                            patch_overlap=patch_overlap,
                            min_labeled_pixels=min_labeled_pixels,
                            num_clusters=num_clusters,
                            voxel_selection=voxel_selection,
                            max_patches=max_patches,
                            model_dir=fe_model_dir,
                            train_input_dir=fe_train_input_dir,
                            pred_input_dir=fe_train_input_dir )
    
    fe.run(batch_size=batch_size, type='training')
    fe.run(batch_size=batch_size, type='prediction')



if classify:
    svm = SvmClassifier(    train_feature_dir=svm_train_feature_dir,
                            train_target_dir=svm_train_target_dir,
                            pred_feature_dir=svm_pred_feature_dir,
                            pred_target_dir=svm_pred_target_dir,
                            ffr_boundary=ffr_boundary   )
                            
    svm.train()
    svm.predict()


if delete_tmp:
    util.delete_tmp_files()
    

