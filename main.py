from clustering import Clustering
from CAE_2D import CAE_2D
from CAE_3D import CAE_3D
from feature_extraction import FeatureExtraction
from svm_classifier import SvmClassifier

#-----------------------------------------------------#
#                   Clustering                        #
#-----------------------------------------------------#
clus_seg_dir = 'data/feature_extraction/segmentation/'
clus_out_dir= 'evaluation/clustering/ex1/'
num_iters = 200
num_clusters = 500

#-----------------------------------------------------#
#             2D Convolutional Autoencoder              #
#-----------------------------------------------------#
cae2d_img_dir = 'data/CAE/imaging/'
cae2d_seg_dir = 'data/CAE/segmentation/'
cae2d_out_dir = 'evaluation/CAE/2D/ex4/'
cae2d_model_dir = 'evaluation/CAE/2D/ex4/'
cae2d_model_name = 'model_2D'
patch_overlap=(0, 46, 46)
min_labeled_pixels=0.7
patch_size_2d = (1, 48, 48)

#-----------------------------------------------------#
#             3D Convolutional Autoencoder              #
#-----------------------------------------------------#
cae3d_img_dir = 'data/CAE/imaging/'
cae3d_seg_dir = 'data/CAE/segmentation/'
cae3d_out_dir = 'evaluation/CAE/3D/ex1/'
cae3d_model_dir = 'evaluation/CAE/3D/ex1/'
cae3d_model_name = 'model_3D'
max_patches = 20
patch_size_3d = (32, 32, 32)

#-----------------------------------------------------#
#               Feature Extraction                    #
#-----------------------------------------------------#
fe_img_dir = 'data/feature_extraction/imaging/'
fe_seg_dir = 'data/feature_extraction/segmentation/'
fe_clus_dir = 'evaluation/clustering/ex1/'
fe_model_dir = 'evaluation/CAE/2D/ex4/'
fe_model_name = 'model_2D'
fe_out_dir = 'evaluation/feature_extraction/ex2/'
#voxel_selection = 'highest_share'
voxel_selection = 'center'

#-----------------------------------------------------#
#               Feature Extraction                    #
#-----------------------------------------------------#
svm_feature_dir = 'evaluation/feature_extraction/ex2/'
svm_out_dir = 'evaluation/svm_classifier/ex1/'
target = [1, 1, 0, 1, 0, 0, 1]



#-----------------------------------------------------#
#               Program Parameters                    #
#-----------------------------------------------------#
batch_size = 500
epochs = 1
batches_per_epoch = 1
test_size = 0.5
load_model_for_pred = False

cluster = False
autoencoder_2d = False
autoencoder_3d = False
extract_features = False
classify = True

#-----------------------------------------------------#
#                  Start of Program                   #
#-----------------------------------------------------#

if cluster:
    clustering = Clustering(    num_iters=num_iters,
                                num_clusters=num_clusters,
                                seg_dir=clus_seg_dir,
                                out_dir=clus_out_dir    )

    clustering.run()


if autoencoder_2d:
    cae = CAE_2D(   model_name=cae2d_model_name,
                    patch_size=patch_size_2d,
                    patch_overlap=patch_overlap,
                    min_labeled_pixels=min_labeled_pixels,
                    test_size=test_size,
                    load_model_for_pred=load_model_for_pred,
                    img_dir=cae2d_img_dir,
                    seg_dir=cae2d_seg_dir,
                    out_dir=cae2d_out_dir, 
                    model_dir=cae2d_model_dir   )
    

    if load_model_for_pred:
        cae.predict(batch_size=batch_size)

    else: 
        cae.train(batch_size=batch_size, epochs=epochs, batches_per_epoch=batches_per_epoch)    
        cae.predict(batch_size=batch_size)


if autoencoder_3d:
    cae = CAE_3D(   model_name=cae3d_model_name,
                    patch_size=patch_size_3d,
                    test_size=test_size,
                    load_model_for_pred=load_model_for_pred,
                    max_patches=max_patches,
                    img_dir=cae3d_img_dir,
                    seg_dir=cae3d_seg_dir,
                    out_dir=cae3d_out_dir, 
                    model_dir=cae3d_model_dir )
    
    cae.train(batch_size=batch_size, epochs=epochs, batches_per_epoch=batches_per_epoch)    



if extract_features:
    fe = FeatureExtraction( model_name=fe_model_name,
                            patch_size=patch_size_2d,
                            patch_overlap=patch_overlap,
                            min_labeled_pixels=min_labeled_pixels,
                            num_clusters=num_clusters,
                            voxel_selection=voxel_selection,
                            max_patches=max_patches,
                            img_dir=fe_img_dir,
                            seg_dir=fe_seg_dir,
                            clus_dir=fe_clus_dir,
                            out_dir=fe_out_dir, 
                            model_dir=fe_model_dir )
    
    fe.run(batch_size=batch_size)


if classify:
    svm = SvmClassifier(    target=target,
                            feature_dir=svm_feature_dir,
                            test_size=test_size,
                            out_dir=svm_out_dir )

    svm.run()
    

