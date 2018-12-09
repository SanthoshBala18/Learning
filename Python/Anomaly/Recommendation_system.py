# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 00:21:38 2018

@author: Sandy
"""
import scipy.io
import pandas as pd
import numpy as np

#Compute the cost function
def compute_cost(x,theta,y,R,l=1.5):
    inner_product = np.matmul(x,theta.transpose())
    inner_product = inner_product*R
    y = y*R
    diff = inner_product-y
    squared_term = np.square(diff)
    sum_squared_term = np.sum(squared_term)
    cost = (1/2)*sum_squared_term+(l/2)*(np.sum(np.square(theta)))+(l/2)*(np.sum(np.square(x)))
  
    x_grad = np.zeros(x.shape)
    for each,row in enumerate(x):
        movie_rated = R[each,:]
        theta_tmp = theta[movie_rated==1]
        y_tmp = Y[each,:]
        y_tmp = y_tmp[movie_rated==1]
        x_row = x[each,:]
        x_row = x_row.reshape((len(x_row),1))
        y_tmp = y_tmp.reshape((len(y_tmp),1))
       
        val = np.matmul(theta_tmp,x_row)-y_tmp
        
        reg = l*x_row
        first_term = np.matmul(val.transpose(),theta_tmp)
        x_row = first_term + reg.transpose()
        x_grad[each,:] = (x_row)
    
    theta_grad = np.zeros(theta.shape)
    for each,row in enumerate(theta):
        user_rated = R[:,each]
        x_tmp = x[user_rated==1]
        y_tmp = Y[:,each]
        y_tmp = y_tmp[user_rated==1]
        theta_row = theta[each,:]
        theta_row = theta_row.reshape((len(theta_row),1))
        y_tmp = y_tmp.reshape((len(y_tmp),1))
        
        val = np.matmul(x_tmp,theta_row)-y_tmp
        
        reg = (l*theta_row)

        theta_row = np.matmul(val.transpose(),x_tmp) + reg.transpose()

        theta_grad[each,:] = (theta_row) 
        
        
    
    return x_grad,theta_grad,cost

#Normalize each column,by subtracting the mean and dividing by standard deviation
def Normalize(X):
    X = pd.DataFrame(X)
    X_norm = X
    mu = np.zeros((1,len(X.columns)))
    sigma = np.zeros((1,len(X.columns)))
    
    for column,each_column in enumerate(X.columns):
        mean = np.mean(X[each_column])
        std = np.std(X[each_column])
        mu[0,column]=mean
        sigma[0,column]=std
        X_norm[each_column] = (X_norm[each_column]-mean)/std
    
    return(X_norm.values)    
#Load the movie data
data = scipy.io.loadmat("ex8_movies")
#print(data.keys()) 
#Load the rating matrix 
Y = data['Y']
#Y = Normalize(Y)
#Y = pd.DataFrame(Y)
#Load the matrix which indicates whether the user has rated the movie or not
R = data['R']
#R = pd.DataFrame(R)
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

X = np.random.rand(num_movies,num_features)
Theta = np.random.rand(num_users,num_features)

for i in range(10):
    x_grad,theta_grad,cost = compute_cost(X,Theta,Y,R,0.01)
    X = X-(1*x_grad/len(X))
    Theta = Theta-(1*theta_grad/len(Theta))
    print(cost)
