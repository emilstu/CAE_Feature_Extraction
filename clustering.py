#https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random
import time
from utils import util
import nibabel as nib
import os

class Clustering:
    def __init__(self, num_iters, num_clusters, seg_dir, out_dir):
        """
        num_iters (int): Number of iterations of kmeans
        num_clusters (int): Number of clusters
        seg_dir (string): direcorty of segmentations to cluster
        out_dir (string): where to store the clustered segmentations
        
        """

        self.num_iters = num_iters
        self.num_clusters = num_clusters
        self.seg_dir = seg_dir
        self.out_dir = out_dir

        # Create out directory for CAE if it doesn't exists 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


    def run(self):
        # Get filenames 
        filenames = util.get_nifti_filenames(self.seg_dir)
        print(filenames)

        # Loop over images
        for i, filename in enumerate(filenames):
            # Print output
            print("###" + str(i)+ ".image ### ")
            print("/n")
            
            img = nib.load(filename)
            img_data = img.get_data()
            
            hd = img.header

            M = img.affine[:3, :3]
            iM = np.linalg.inv(M)

            abc = img.affine[:3, 3]


            # Get coordinate data of image
            data = self.get_coordinates_from_segmentation(img_data, M, iM, abc)
            

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
            for n in range(self.num_iters):
                # Get closest centroids to each data point
                assigned_centroids = self.get_closest_centroid(data[:, None, :], centroids[None,:, :])    

                # Compute new centroids
                for c in range(centroids.shape[1]):
                    # Get data points belonging to each cluster 
                    cluster_members = data[assigned_centroids == c]
                    
                    # Compute the mean of the clusters
                    cluster_members = cluster_members.mean(axis = 0)
                    
                    # Update the centroids
                    centroids[c] = cluster_members

                # Compute SSE
                sse = self.compute_sse(data.squeeze(), centroids.squeeze(), assigned_centroids)
                sse_list.append(sse)

            # End time
            toc = time.time()

            # Print output
            print("-- Total clustering-time: " + str(round(toc - tic, self.num_clusters)))
            print("-- Clustering-time per iteration: " + str(round(toc - tic, self.num_clusters)/self.num_iters))
            print("\n")
            print("\n")
            print("\n")
    
    
            # Save clustering to nifti label-map
            label_map = np.zeros_like(img_data)

            label = 1
            for c in range(len(centroids)):
                cluster_members = [data[i] for i in range(len(data)) if assigned_centroids[i] == c]    
                cluster_members = np.array(cluster_members)
                
                for cord in cluster_members:
                    ijk = self.invf(cord[0], cord[1], cord[2], iM, abc)
                    label_map[int(ijk[0]),int(ijk[1]),int(ijk[2])] = label
                
                label+=1
            
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

            outpath = f'{self.out_dir}label_map_{i}.nii.gz'
            nib.save(label_map, outpath)
    

    
    def get_closest_centroid(self, x, centroids):
        # Loop over each centroid and compute the distance from data point.
        dist = self.compute_l2_distance(x, centroids)

        # Get the index of the centroid with the smallest distance to the data point 
        closest_centroid_index =  np.argmin(dist, axis = 1)
        
        return closest_centroid_index

    def compute_sse(self, data, centroids, assigned_centroids):
        # Initialise SSE 
        sse = 0

        # Compute SSE
        sse = self.compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
        
        return sse


    def compute_l2_distance(self, x, centroid):
        # Compute the difference, following by raising to power 2 and summing
        dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
        return dist

    def f(self, i, j, k, M, abc):
        """ Return X, Y, Z coordinates for i, j, k """
        return M.dot([i, j, k]) + abc

    def invf(self, x, y, z, iM, abc):
        """ Return i, j, k indexes for x, y, z """
        return (iM.dot([x, y, z] - abc))


    def get_coordinates_from_segmentation(self, image_data, M, iM, abc):
        
        # Get indices 
        n_i, n_j, n_k = image_data.shape

        #Get coordinate array from 1-voxels
        xyz_array = []

        start_time = time.time()
        for i in range(0, n_i):
            for j in range(0, n_j):
                for k in range(0, n_k):
                    if image_data[i,j,k] == 1:
                        xyz = self.f(i, j, k, M, abc)
                        xyz_array.append(xyz)
                
        end_time = time.time()
        total_duration = end_time - start_time
        print("-- Time used to get coordinate matrix: " + str(round(total_duration,4)))
        
        xyz_data = np.asarray(xyz_array)

        return(xyz_data)