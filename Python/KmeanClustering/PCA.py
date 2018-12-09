# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:46:21 2018

@author: Sandy
"""

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import cv2

#Normalize each column,by subtracting the mean and dividing by standard deviation
def Normalize(X):
    X_norm = X
    mu = np.zeros((1,len(X.columns)))
    sigma = np.zeros((1,len(X.columns)))
    
    for column,each_column in enumerate(X.columns):
        mean = np.mean(X[each_column])
        std = np.std(X[each_column])
        mu[0,column]=mean
        sigma[0,column]=std
        X_norm[each_column] = (X_norm[each_column]-mean)/std
    
    return(X_norm)

#Compute the covariance matrix
def compute_Covariance(X):
    x_array = X.values
    covariance_matrix = (1/len(x_array))*(np.matmul(x_array.transpose(),x_array))
    
    return covariance_matrix

#Compute the eigen vectors using singular value decompositions
def compute_SVD(x):
    u,s,v = np.linalg.svd(x)
    
    return u

#Project the data onto lower dimension
def project_data(X,u,k=1):
    u_df = pd.DataFrame(u)
    first_k_components = u_df.iloc[:,0:(k)]
    
    first_k_components = first_k_components.values
    
    u_reduce = np.matmul(first_k_components.transpose(),X.transpose())
    
    return (u_reduce.transpose())

#Recover data from lower to higher dimension
def recover_data(z,u,k=1):
    u_df = pd.DataFrame(u)
    first_k_components = u_df.iloc[:,0:(k)]
    first_k_components = first_k_components.values
    z = z.reshape(len(z),k)
    first_k_components = first_k_components.reshape(len(first_k_components),k)
    print(z.shape)
    print(first_k_components.shape)
    recovered_x = np.matmul(z,first_k_components.transpose())
    
    return(recovered_x)

def Example():
    data = scipy.io.loadmat("ex7data1")
    X = data['X']
    X = pd.DataFrame(X)
    X_norm = Normalize(X)
    
    covariance_matrix = compute_Covariance(X_norm)
    principal_components = compute_SVD(covariance_matrix)
    u_reduce = project_data(X_norm,principal_components)
    recover_data(u_reduce,principal_components)
   
#Load the data containing images
data = scipy.io.loadmat("ex7faces")
face_images = data['X']
X = pd.DataFrame(face_images)
X_norm = Normalize(X)
#Compute the principal components
covariance_matrix = compute_Covariance(X_norm)
principal_components = compute_SVD(covariance_matrix)
#Project the data onto the lower dimensions
u_reduce = project_data(X_norm,principal_components,100)
x_recovered = recover_data(u_reduce,principal_components,100)
#Display the first original and reconstructed image
first_image = X_norm.values[0,:]
first_image_reduced = x_recovered[0,:]
cv2.imshow('original',first_image.reshape(32,32))
cv2.imshow('Reconstructed',first_image_reduced.reshape(32,32))
cv2.waitKey(0)
cv2.destroyAllWindows()