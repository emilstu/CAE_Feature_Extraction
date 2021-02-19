#https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import time
from utils import util
import nibabel as nib
import os
from tqdm import tqdm
from utils.kmeans import get_closest_centroid, compute_sse, get_coordinates_from_segmentation, invf

class Clustering:
    def __init__(self, num_iters, num_clusters, train_input_dir, pred_input_dir):
        """
        num_iters (int): Number of iterations of kmeans
        num_clusters (int): Number of clusters
        seg_dir (string): direcorty of segmentations to cluster
        out_dir (string): where to store the clustered segmentations
        
        """

        self.num_iters = num_iters
        self.num_clusters = num_clusters
        self.train_input_dir = train_input_dir
        self.pred_input_dir = pred_input_dir
        

    def run(self, type):
        # Get filenames 
        if type == 'training':
            filenames = util.get_paths_from_tree(self.train_input_dir, 'segmentation')
            print('\n\nRunning clustering on training data...\n\n')  
        elif type == 'prediction':
            filenames = util.get_paths_from_tree(self.train_input_dir, 'segmentation')
            print('\n\nRunning clustering on prediction data...\n\n')  
        else:
            raise Exception('\nException: Clustering type must either be "training" or "prediction"\n') 

        # Loop over images
        for i, filename in enumerate(tqdm(filenames)):
            # Print output
            
            img = nib.load(filename)
            img_data = img.get_data()
            
            hd = img.header

            M = img.affine[:3, :3]
            iM = np.linalg.inv(M)

            abc = img.affine[:3, 3]


            # Get coordinate data of image
            print('\n\nGet coordinate matrix from segmentation.. ')
            data = get_coordinates_from_segmentation(img_data, M, iM, abc)
            

            # Shuffle the data
            #np.random.shuffle(data)

            # Set random seed for reproducibility 
            random.seed(0)

            # Initialise centroids
            centroids = data[random.sample(range(data.shape[0]), self.num_clusters)]

            # Create a list to store which centroid is assigned to each dataset
            assigned_centroids = np.zeros(len(data), dtype = np.int32)

            # Number of dimensions in centroid
            num_centroid_dims = data.shape[1]

            # List to store SSE for each iteration 
            sse_list = []

            # Start time
            tic = time.time()

            # Main Loop
            print('\nStart clustering image...')
            for n in tqdm(range(self.num_iters), leave=False):
                # Get closest centroids to each data point
                assigned_centroids = get_closest_centroid(data[:, None, :], centroids[None,:, :])    

                # Compute new centroids
                for c in range(centroids.shape[1]):
                    # Get data points belonging to each cluster 
                    cluster_members = data[assigned_centroids == c]
                    
                    # Compute the mean of the clusters
                    cluster_members = cluster_members.mean(axis = 0)
                    
                    # Update the centroids
                    centroids[c] = cluster_members

                # Compute SSE
                sse = compute_sse(data.squeeze(), centroids.squeeze(), assigned_centroids)
                sse_list.append(sse)
            print('Finished clustering image.')

            # End time
            toc = time.time()

            # Print output
            print("Total clustering-time: " + str(round(toc - tic, self.num_clusters)))
            print("Clustering-time per iteration: \n" + str(round(toc - tic, self.num_clusters)/self.num_iters))
        
    
            # Save clustering to nifti label-map
            label_map = np.zeros_like(img_data)

            label = 1

            print('\nSaving clustered image to nifti label-map... \n')
            for c in tqdm(range(len(centroids)), leave=False):
                cluster_members = [data[i] for i in range(len(data)) if assigned_centroids[i] == c]    
                cluster_members = np.array(cluster_members)
                
                for cord in cluster_members:
                    ijk = invf(cord[0], cord[1], cord[2], iM, abc)
                    label_map[int(ijk[0]),int(ijk[1]),int(ijk[2])] = label
                
                label+=1

            folder_name = util.get_sub_dirs(self.train_input_dir)[i]
            self.save_clustered_image(img, label_map, hd, folder_name, type)
            print('File saved.')


    def save_clustered_image(self, img, label_map, hd, folder_name, type):    
        # update data type:
        new_dtype = np.float32 
        label_map = label_map.astype(new_dtype)
        img.set_data_dtype(new_dtype)
        
        # if nifty1
        if hd['sizeof_hdr'] == 348:
            label_map = nib.Nifti1Image(label_map, img.affine, header=hd)
        # if nifty2
        elif hd['sizeof_hdr'] == 540:
            label_map = nib.Nifti2Image(label_map, img.affine, header=hd)
        else:
            raise IOError('Input image header problem')

        
        if type == 'training':
            outpath = f'{self.train_input_dir}{folder_name}/cluster.nii.gz' 
        else:
            outpath = f'{self.pred_input_dir}{folder_name}/cluster.nii.gz' 
            
        nib.save(label_map, outpath)
    

    