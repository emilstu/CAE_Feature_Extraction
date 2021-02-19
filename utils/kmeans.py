import numpy as np
import time
from tqdm import tqdm

def get_closest_centroid(x, centroids):
        # Loop over each centroid and compute the distance from data point.
        dist = compute_l2_distance(x, centroids)

        # Get the index of the centroid with the smallest distance to the data point 
        closest_centroid_index =  np.argmin(dist, axis = 1)
        
        return closest_centroid_index

def compute_sse(data, centroids, assigned_centroids):
    # Initialise SSE 
    sse = 0

    # Compute SSE
    sse = compute_l2_distance(data, centroids[assigned_centroids]).sum() / len(data)
    
    return sse


def compute_l2_distance(x, centroid):
    # Compute the difference, following by raising to power 2 and summing
    dist = ((x - centroid) ** 2).sum(axis = x.ndim - 1)
    return dist

def f(i, j, k, M, abc):
    """ Return X, Y, Z coordinates for i, j, k """
    return M.dot([i, j, k]) + abc

def invf(x, y, z, iM, abc):
    """ Return i, j, k indexes for x, y, z """
    return (iM.dot([x, y, z] - abc))


def get_coordinates_from_segmentation(image_data, M, iM, abc):
    
    # Get indices 
    n_i, n_j, n_k = image_data.shape

    #Get coordinate array from 1-voxels
    xyz_array = []

    start_time = time.time()
    for i in tqdm(range(0, n_i), leave=False):
        for j in range(0, n_j):
            for k in range(0, n_k):
                if image_data[i,j,k] == 1:
                    xyz = f(i, j, k, M, abc)
                    xyz_array.append(xyz)
            
    end_time = time.time()
    total_duration = end_time - start_time
    print("\nTime used to get coordinate matrix: " + str(round(total_duration,4)))
    
    xyz_data = np.asarray(xyz_array)

    return(xyz_data)