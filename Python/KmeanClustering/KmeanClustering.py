# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:01:19 2018

@author: Sandy
"""
import scipy.io
import numpy as np
import pandas as pd
import cv2

#Square of a number
def square(n):
    return (n*n)
   
#Squared distance between two vectors     
def findDistance(x,y):
    diff = x-y
    distance = 0
    for each in diff:
        distance = distance+square(each)
    return distance

#function to find the closest cluster for each of the data point
def findClosestCentroids(X,centroids):
    c = []
    for pos,each in enumerate(X):
        minDist = 0
        c.append(0)
        for idx,centroid in enumerate(centroids):
            centroid_pos = idx+1
            dist = findDistance(each,centroid)
            if (dist<=minDist or minDist==0):
                minDist = dist
                c[pos] = centroid_pos
                
    return c

#Function to compute the centroid for each cluster
def computeCentroid(X,idx=0,K=0):
   array = pd.Series(idx)
   
   df = pd.DataFrame(X)
   df = pd.concat([df,array],ignore_index=True,axis=1)
   last_col = len(df.columns)-1
   centroid_group_mean = df.groupby(last_col).mean()
   centroid = np.ndarray((K,len(centroid_group_mean.columns)))
   
   for each_row in range(K):
       for each_column in range(len(centroid_group_mean.columns)):
           centroid[each_row][each_column] = centroid_group_mean.loc[(each_row+1),each_column] 
           
   return(centroid)

#Function to compute the K mean algorithm
def run_Kmeans(X,initial_centroids,max_iters=10):
    centroids = initial_centroids
    K = len(initial_centroids)
    
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroid(X,idx,K)
    
    return (centroids)
    
#Sample run with sample 2D data Set
def Example():
    initial_centroids = np.ndarray((3,2))
    initial_centroids[0][0] = 3
    initial_centroids[0][1] = 3
    initial_centroids[1][0] = 6
    initial_centroids[1][1] = 2
    initial_centroids[2][0] = 8
    initial_centroids[2][1] = 5
    
    data = scipy.io.loadmat("ex7data2")
    X = data['X']

    run_Kmeans(X,initial_centroids,6)

def recover_image(centroids,idx):
    recovered_image = np.ndarray((len(idx),len(centroids[0])))
    
    for pos,each in enumerate(idx):
        recovered_image[pos] = centroids[each-1]
    return recovered_image

#Load the image
image = cv2.imread("bird_small.png",cv2.IMREAD_COLOR)
original_image = image

#Normalize the image so that all values range between 0 and 1
original_image_normalized = original_image/255
origial_shape = original_image_normalized.shape
#Reshape it into 2D array
pixel_matrix = original_image_normalized.reshape((origial_shape[0]*origial_shape[1]),3)

#Initialize centroid
initial_centroids = pixel_matrix[0:16]

#Run K means to find K centroids
centroids = run_Kmeans(pixel_matrix,initial_centroids,max_iters=10)

#Find the closest of each in the final centroid
idx = findClosestCentroids(pixel_matrix,centroids)

#Recover the original image with only the cluster centroids found
X_recovered = recover_image(centroids,idx)
X_recovered = X_recovered.reshape(origial_shape)

cv2.imshow('original',original_image)
cv2.imshow('Reconstructed',X_recovered)
cv2.waitKey(0)
cv2.destroyAllWindows()




















