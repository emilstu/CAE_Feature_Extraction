from tensorflow.python.ops.gen_batch_ops import batch
#from automatic_segmentation import AutomaticSegmentation
from clustering import Clustering
from feature_extraction import FeatureExtraction
from svm_classifier import SvmClassifier
import utils.argparser_utils as apu
from CAE.cae import CAE
import json
from utils import util
import argparse
import sys
import ast


def main(args):
    #-----------------------------------------------------#
    #             2D/3D Convolutional Autoencoder         #
    #-----------------------------------------------------#
    if args.program=='CAE':
        cae = CAE(  input_dir=args.data_dir,
                    patch_size=ast.literal_eval(args.patch_size),
                    batch_size=args.batch_size,
                    test_size=args.test_size, 
                    prepare_batches=args.prepare_batches )
        
        cae.prepare_data(   args.sampler_type,
                            args.max_patches,
                            args.resample,
                            ast.literal_eval(args.patch_overlap),
                            args.min_lab_vox,
                            args.label_prob,
                            args.load_data   )
        if args.model_dir is None:
            cae.train(args.epochs)
        cae.predict(args.model_dir)

    #-----------------------------------------------------#
    #               Patient classification                #
    #-----------------------------------------------------#
    """
    if args.program=='AutSeg':
        asg = AutomaticSegmentation(    model_name=args.model_name,
                                        patch_size=args.patch_size,
                                        patch_overlap=args.patch_overlap,
                                        input_dir=args.data_dir, 
                                        model_dir=args.model_dir   )
        asg.run()
        asg.run_postprocessing()

"""
    if args.program=='CLUS':
        clustering = Clustering(    num_iters=args.iterations,
                                    num_clusters=args.num_clusters,
                                    input_dir=args.data_dir  )
        clustering.run() 


    if args.program=='FeEx':
        fe = FeatureExtraction( model_name=args.model_name,
                                patch_size=ast.literal_eval(args.patch_size),
                                patch_overlap=ast.literal_eval(args.patch_overlap),
                                num_clusters=args.num_clusters,
                                cluster_selection=args.cluster_selection,
                                resample=args.resample,
                                encoded_layer_num=args.encoded_layer_num,
                                model_dir=args.model_dir,
                                input_dir=args.data_dir )
        fe.run(batch_size=20)


    if args.program=='SVM':
        svm = SvmClassifier(    feature_dir=args.feature_dir,
                                ffr_dir=args.ffr_dir,
                                ffr_filename=args.ffr_filename,
                                input_dir=args.data_dir,
                                ffr_cut_off=args.ffr_cut_off,   
                                test_size=args.test_size  )                  
        svm.train()
        svm.predict()

    
if __name__ == '__main__':
    command_parser = argparse.ArgumentParser(add_help=False)
    command_parser.add_argument('-h', '--help', action=apu._HelpAction, help='Choose program to run')
    sub_parser = command_parser.add_subparsers(help='---- Program ----', dest='program')
    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=52)
    #-----------------------------------------------------#
    #             2D/3D Convolutional Autoencoder         #
    #-----------------------------------------------------#
    cae_parser = sub_parser.add_parser('CAE', help='3D/2D Convolutional Autoencoder', formatter_class=formatter)
    cae_parser.add_argument('-data_dir', required=False, default='data/CAE/patients/', type=str,
                            help='Directory where data is stored')
    cae_parser.add_argument('-patch_size', required=True, type=str,
                            help='Patch size 2D/3D: "(1,int,int)" or "(int,int,int)"')
    cae_parser.add_argument('-patch_overlap', required=True, type=str, 
                            help='Patch overlap 2D/3D: (0,int,int) or (int,int,int). Must be even number and smaller than patch size')
    cae_parser.add_argument('-sampler_type', required=True, choices=['grid', 'label'],
                            help='Sampler type')
    cae_parser.add_argument('-min_lab_vox', required=False, default=1.0, type=apu.check_range, 
                            help='Minimum labled voxels used by grid-sampler')
    cae_parser.add_argument('-label_prob', required=False, default=1.0, type=float,
                            help='Probability of choosing patches with labeled voxel as center. Used by label-sampler')
    cae_parser.add_argument('-max_patches', required=False, default=None, type=int,
                            help='Maximum number of patches to extract')
    cae_parser.add_argument('-resample', required=False, default=(), type=str,
                             help='Resample to common voxel spacing (float,float,float)')
    cae_parser.add_argument('-epochs', required=False, default=500, type=int, 
                            help='Number of epochs in training of CAE')
    cae_parser.add_argument('-batch_size', required=False, default=500, type=int, 
                            help='Batch size for training')
    cae_parser.add_argument('-test_size', required=False, default=0.2, type=apu.check_range, 
                            help='CAE test size. Float between 0.0 and 1.0')
    cae_parser.add_argument('-prepare_batches', required=False,  default=False, dest='prepare_batches', action='store_true',
                            help='Specified if batches should be prepared and saved in mini-batches')
    cae_parser.add_argument('-load_data', required=False, default=False, dest='load_data', action='store_true',
                            help='Specified if patches sould be loaded. For this option to work data must exist in the tmp folder')
    cae_parser.add_argument('-model_dir', required=False, default=None, type=str,
                            help='Directory of model if model should be loaded for prediction')

    #-----------------------------------------------------#
    #               Automatic Segmentation                #
    #-----------------------------------------------------#
    ase_parser = sub_parser.add_parser('AutSeg', help='Automatic segmentation of LV-myocardium', formatter_class=formatter)
    ase_parser.add_argument('-data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored')
    ase_parser.add_argument('-model_dir', required=True, type=str,
                            help='Directory where model is stored')
    ase_parser.add_argument('-model_name', required=True, type=str,
                            help='Model name, i.e. "model.best"')
    ase_parser.add_argument('-patch_size', required=True, type=str,
                            help='Patch size used when the model was trained: "(int,int,int)"')
    
    #-----------------------------------------------------#
    #                     Clustering                      #
    #-----------------------------------------------------#
    clus_parser = sub_parser.add_parser('CLUS', help='Clustering of LV-myocardium', formatter_class=formatter)
    clus_parser.add_argument('-data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored')
    clus_parser.add_argument('-iterations', required=True, type=int,
                            help='Number of iterations to run-kmeans clustering')
    clus_parser.add_argument('-num_clusters', required=True, type=int,
                            help='Number of clusters')

    #-----------------------------------------------------#
    #               2D/3D Feature Extraction              #
    #-----------------------------------------------------#
    fe_parser = sub_parser.add_parser('FeEx', help='3D/2D FeatureExtraction', formatter_class=formatter)
    fe_parser.add_argument('-data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored')
    fe_parser.add_argument('-model_dir', required=True, type=str,
                            help='Directory where model is stored')
    fe_parser.add_argument('-model_name', required=True, type=str,
                            help='Model name, i.e. "model_2D"')
    fe_parser.add_argument('-patch_size', required=True, type=str,
                            help='Patch size 3D/3D: "(1,int,int)" or "(int,int,int)"')
    fe_parser.add_argument('-patch_overlap', required=False, type=str, 
                            help='Patch overlap 2D/3D: "(0,int,int)" or "(int,int,int)". Must be even number and smaller than patch size')
    fe_parser.add_argument('-cluster_selection', required=False, default='center', choices=['center', 'highest_share'], type=str,
                            help='Method used to select which cluster a specific patch belongs to')
    fe_parser.add_argument('-num_clusters', required=True, type=int,
                            help='Number og clusters used in the images to extract features from')
    fe_parser.add_argument('-resample', required=False, default=(), type=str,
                             help='Resample to common voxel spacing (float,float,float)')
    fe_parser.add_argument('-encoded_layer_num', required=True, type=int,
                            help='Number of the encoded layer counting from the bottom')

    #-----------------------------------------------------#
    #                   SVM Classification                #
    #-----------------------------------------------------#
    svm_parser = sub_parser.add_parser('SVM', help='SVM-Classification of extracted festures', formatter_class=formatter)
    svm_parser.add_argument('-data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored')
    svm_parser.add_argument('-feature_dir', required=True, type=str,
                            help='Directory where features are stored, i.e. output from FeEx')
    svm_parser.add_argument('-ffr_dir', required=False, default='data/classification/ffr_data/', type=str,
                            help='Directory ffr_values are stores')
    svm_parser.add_argument('-ffr_filename', required=True, type=str,
                            help='Filename for the file where ffr-values are stored')
    svm_parser.add_argument('-ffr_cut_off', required=False, default=0.85, type=str,
                            help='Filename for the file where ffr-values are stored')
    svm_parser.add_argument('-test_size', required=False, default=0.4, type=apu.check_range, 
                            help='SVM test size. Float between 0.0 and 1.0')
    try:
        args = command_parser.parse_args()
    except:
        command_parser.print_help()
        sys.exit(0)

    main(args)
    
    