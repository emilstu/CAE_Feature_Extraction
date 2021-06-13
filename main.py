from tensorflow.python.ops.gen_batch_ops import batch
from automatic_segmentation import AutomaticSegmentation
from clustering import Clustering
from feature_extraction import FeatureExtraction
from classifier import Classifier
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
        
        cae.prepare_data(   sampler_type=args.sampler_type,
                            max_patches=args.max_patches,
                            resample=ast.literal_eval(args.resample),
                            clipping=ast.literal_eval(args.clipping),
                            pixel_norm=args.pixel_norm,
                            min_labeled_voxels=args.min_lab_vox,
                            patch_overlap=ast.literal_eval(args.patch_overlap),
                            label_prob=args.label_prob,
                            load_data=args.load_data)   
        if args.model_dir is None:
            cae.train(args.epochs)
        cae.predict(args.model_dir)

    #-----------------------------------------------------#
    #               Patient classification                #
    #-----------------------------------------------------#
    if args.program=='AUS':
        asg = AutomaticSegmentation(    model_name=args.model_name,
                                        patch_size=ast.literal_eval(args.patch_size),
                                        patch_overlap=ast.literal_eval(args.patch_overlap),
                                        input_dir=args.data_dir, 
                                        model_dir=args.model_dir   )
        asg.run()
        asg.run_postprocessing()

    if args.program=='CLUS':
        clustering = Clustering(    num_iters=args.iterations,
                                    num_clusters=args.num_clusters,
                                    input_dir=args.data_dir  )
        clustering.run() 


    if args.program=='FEX':
        fe = FeatureExtraction( model_name=args.model_name,
                                patch_size=ast.literal_eval(args.patch_size),
                                patch_overlap=ast.literal_eval(args.patch_overlap),
                                min_labeled_voxels=args.min_lab_vox,
                                num_clusters=args.num_clusters,
                                cluster_selection=args.cluster_selection,
                                resample=ast.literal_eval(args.resample),
                                clipping=ast.literal_eval(args.clipping),
                                pixel_norm=args.pixel_norm,
                                encoded_layer_num=args.encoded_layer_num,
                                save_patches=args.save_patches,
                                model_dir=args.model_dir,
                                input_dir=args.data_dir,
                                spn=args.start_patient_num  )
        fe.run(batch_size=50)


    if args.program=='CLA':
        clas = Classifier(  feature_dir=args.feature_dir,
                            ffr_dir=args.ffr_dir,
                            ffr_filename=args.ffr_filename,
                            input_dir=args.data_dir,
                            ffr_cut_off=args.ffr_cut_off,   
                            folds=args.folds,  
                            iterations=args.iterations  )           
        clas.train(args.feature_selection, args.plot, args.chi2, args.mutual_information)

    
if __name__ == '__main__':
    command_parser = argparse.ArgumentParser(add_help=False)
    command_parser.add_argument('-h', '--help', action=apu._HelpAction, help='Choose program to run')
    sub_parser = command_parser.add_subparsers(help='---- Program ----', dest='program')
    formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=80)
    #-----------------------------------------------------#
    #             2D/3D Convolutional Autoencoder         #
    #-----------------------------------------------------#
    cae_parser = sub_parser.add_parser('CAE', help='3D/2D Convolutional Autoencoder', formatter_class=formatter)
    cae_parser.add_argument('-dd','--data_dir', required=False, default='data/CAE/patients/', type=str,
                            help='Directory where data is stored.')
    cae_parser.add_argument('-ps', '--patch_size', required=True, type=str,
                            help='Patch size 2D/3D: "(1,int,int)" or "(int,int,int)".')
    cae_parser.add_argument('-po', '--patch_overlap', required=True, type=str, 
                            help='Patch overlap 2D/3D: (0,int,int) or (int,int,int). Must be even number and smaller than patch size.')
    cae_parser.add_argument('-st','--sampler_type', required=True, choices=['grid', 'label'],
                            help='Sampler type')
    cae_parser.add_argument('-mlv','--min_lab_vox', required=False, default=1.0, type=apu.check_range, 
                            help='Minimum labled voxels used by grid-sampler.')
    cae_parser.add_argument('-lb','--label_prob', required=False, default=1.0, type=float,
                            help='Probability of choosing patches with labeled voxel as center. Used by label-sampler.')
    cae_parser.add_argument('-mp','--max_patches', required=False, default=None, type=int,
                            help='Maximum number of patches to extract.')
    cae_parser.add_argument('-r','--resample', required=False, default=str(()), type=str,
                             help='Resample to common voxel spacing "(float,float,float)".')
    cae_parser.add_argument('-c', '--clipping', required=False, type=str, default=str(()),
                            help='Clipping range: "(int, int)"')
    cae_parser.add_argument('-pn','--pixel_norm', required=False, choices=['z-score', 'minmax', 'abs'], default='minmax',
                            help='Pixel normalization type')
    cae_parser.add_argument('-e','--epochs', required=False, default=500, type=int, 
                            help='Number of epochs in training of CAE.')
    cae_parser.add_argument('-bs','--batch_size', required=False, default=500, type=int, 
                            help='Batch size for training.')
    cae_parser.add_argument('-ts','--test_size', required=False, default=0.2, type=apu.check_range, 
                            help='CAE test size. Float between 0.0 and 1.0.')
    cae_parser.add_argument('-pb','--prepare_batches', required=False,  default=False, dest='prepare_batches', action='store_true',
                            help='Specified when batches should be prepared and saved as mini-batches.')
    cae_parser.add_argument('-ld','--load_data', required=False, default=False, dest='load_data', action='store_true',
                            help='Specified when patches should be loaded.')
    cae_parser.add_argument('-md','--model_dir', required=False, default=None, type=str,
                            help='Directory where model is stored. When specified predictions are made on the loaded model.')

    #-----------------------------------------------------#
    #               Automatic Segmentation                #
    #-----------------------------------------------------#
    ase_parser = sub_parser.add_parser('AUS', help='Automatic segmentation of LV-myocardium', formatter_class=formatter)
    ase_parser.add_argument('-dd', '--data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored.')
    ase_parser.add_argument('-md', '--model_dir', required=False, default='data/classification/as_model/', type=str,
                            help='Directory where model is stored.')
    ase_parser.add_argument('-mn', '--model_name', required=True, type=str,
                            help='Model name, i.e. "model.best".')
    ase_parser.add_argument('-ps', '--patch_size', required=False, type=str, default='(0,0,0)',
                            help='Patch size used when the model was trained: "(int,int,int)".')
    ase_parser.add_argument('-po', '--patch_overlap', required=False, type=str, default='(0,0,0)',
                            help='Patch overlap: "(int,int,int)".')
                 
    #-----------------------------------------------------#
    #                     Clustering                      #
    #-----------------------------------------------------#
    clus_parser = sub_parser.add_parser('CLUS', help='Clustering of LV-myocardium', formatter_class=formatter)
    clus_parser.add_argument('-dd', '--data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored.')
    clus_parser.add_argument('-i','--iterations', required=True, type=int,
                            help='Number of iterations to run k-means clustering.')
    clus_parser.add_argument('-nc', '--num_clusters', required=True, type=int,
                            help='Number of clusters.')

    #-----------------------------------------------------#
    #               2D/3D Feature Extraction              #
    #-----------------------------------------------------#
    fe_parser = sub_parser.add_parser('FEX', help='3D/2D Feature Extraction', formatter_class=formatter)
    fe_parser.add_argument('-dd', '--data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored.')
    fe_parser.add_argument('-md', '--model_dir', required=True, type=str,
                            help='Directory where model is stored.')
    fe_parser.add_argument('-mn', '--model_name', required=True, type=str,
                            help='Model name, i.e. "model_2D".')
    fe_parser.add_argument('-ps', '--patch_size', required=True, type=str,
                            help='Patch size 3D/3D: "(1,int,int)" or "(int,int,int)".')
    fe_parser.add_argument('-po', '--patch_overlap', required=False, default='(0,0,0)', type=str, 
                            help='Patch overlap 2D/3D: "(0,int,int)" or "(int,int,int)". Must be even number and smaller than patch size.')
    fe_parser.add_argument('-mlv','--min_lab_vox', required=False, default=1.0, type=apu.check_range, 
                            help='Minimum labled voxels used by grid-sampler.')
    fe_parser.add_argument('-cs', '--cluster_selection', required=False, default='center', choices=['center', 'highest_share'], type=str,
                            help='Method used to select which cluster a specific patch belongs to.')
    fe_parser.add_argument('-nc', '--num_clusters', required=True, type=int,
                            help='Number og clusters used in the images to extract features from.')
    fe_parser.add_argument('-r','--resample', required=False, default=str(()), type=str,
                             help='Resample to common voxel spacing "(float,float,float)".')
    fe_parser.add_argument('-c', '--clipping', required=False, type=str, default=str(()),
                            help='Clipping range: "(int, int)"')
    fe_parser.add_argument('-pn','--pixel_norm', required=False, choices=['z-score', 'minmax', 'abs'], default='minmax',
                            help='Pixel normalization type')
    fe_parser.add_argument('-eln','--encoded_layer_num', required=True, type=int,
                            help='Number of the encoded layer in CAE-architecture counting from the bottom.')
    fe_parser.add_argument('-spn','--start_patient_num', required=False, type=int, default=0,
                            help='Starting point of the patient in patient data dir.')
    fe_parser.add_argument('-sp','--save_patches', required=False,  default=False, dest='save_patches', action='store_true',
                            help='Specified if patches should be saved to disk.')

    #-----------------------------------------------------#
    #                    Classification                   #
    #-----------------------------------------------------#
    cla_parser = sub_parser.add_parser('CLA', help='Classification of extracted festures', formatter_class=formatter)
    cla_parser.add_argument('-dd', '--data_dir', required=False, default='data/classification/patients/', type=str,
                            help='Directory where data is stored.')
    cla_parser.add_argument('-fd', '--feature_dir', required=True, type=str,
                            help='Directory where features are stored, i.e. output from FEX.')
    cla_parser.add_argument('-ffd','--ffr_dir', required=False, default='data/classification/ffr_data/', type=str,
                            help='Directory ffr_values are stores.')
    cla_parser.add_argument('-ffn','--ffr_filename', required=True, type=str,
                            help='Filename for the file where ffr-values are stored.')
    cla_parser.add_argument('-ffc','--ffr_cut_off', required=False, default=0.8, type=apu.check_range,
                            help='Filename for the file where ffr-values are stored.')
    cla_parser.add_argument('-f','--folds', required=False, default=5, type=int, 
                            help='Number of folds each iteration in cross-validation.')
    cla_parser.add_argument('-i','--iterations', required=False, default=20, type=int, 
                            help='Number of iterations to run cross-validation.')
    cla_parser.add_argument('-fs', '--feature_selection', required=False, type=str, choices=['train', 'entire'], default='train',
                            help='Specification of FS should be performed on train set vs entire dataset')
    cla_parser.add_argument('-c2', '--chi2', required=False, default='all',
                            help='kbest chi2. Default setting is set to all')
    cla_parser.add_argument('-mi', '--mutual_information', required=False, default='all',
                            help='kbest mutual information. Default is set to all')
    cla_parser.add_argument('-plt', '--plot', required=False, default=False, action='store_true',
                            help='Specified if feature statistics (chi2 and MI) should be plotted')
    try:
        args = command_parser.parse_args()
    except:
        command_parser.print_help()
        sys.exit(0)

    main(args)
    
    