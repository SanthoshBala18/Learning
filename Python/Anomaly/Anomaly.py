# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 21:36:23 2018

@author: Sandy
"""

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Find the mean and standard deviation of each feature
def find_mean_and_variance(X):
 
    mu = np.ndarray((1,len(X.columns)))
    var = np.ndarray((1,len(X.columns)))

    for column,each_column in enumerate(X.columns):
        mean = np.mean(X[each_column])
        variance = np.var(X[each_column])
        mu[0,column]=mean
        var[0,column]=variance
    
    return(mu,var)
    
#Compute Gaussian Distribution
def Gaussian(x,mu,sigma):
    n = mu.shape[1]
   
    mu = mu.flatten()
    sigma = sigma.flatten()
    diff_matrix =  x-mu
    covariance_matrix = np.diag(sigma)
    det = np.linalg.det(covariance_matrix)
    gaussian_probability = np.ndarray((len(x),1))
    
    denominator = ((2*np.pi)**(n/2))*((det)**(1/2))
    
    for rowNum,each in enumerate(diff_matrix):    
        power_term = -1/2*(np.matmul(np.matmul(each.transpose(),np.linalg.inv(covariance_matrix)),each))
        exp_term = np.exp(power_term)
        gaussian_probability[rowNum] = exp_term/denominator

    return(gaussian_probability)

#Selecting the threshold value to categorize as anomaly
def select_threshold(p,y):
    eps = 8.3233400720396344e-05
    prob = np.ndarray((len(p),1))
    
    for rownum,each in enumerate(p):
        if each>eps:
            prob[rownum] = 0
        else:
            prob[rownum] = 1
    #True positives
    tp = np.sum((y==1) & (prob==1))
    #False positive
    fp = np.sum((y==0) & (prob==1))
    #False Negative
    fn = np.sum((y==1) & (prob==0))
    
    #Precision
    precision = (tp)/(tp+fp)
    #Recall
    recall = (tp)/(tp+fn)
    #F1 Score
    f1_score = (2*precision*recall)/(precision+recall)
    print(f1_score)
   
    
#Load the two parameter data
data = scipy.io.loadmat("ex8data1") 
X = data['X']
Xval = data['Xval']
yval = data['yval']
#Plot the data
#plt.scatter(X[:,0],X[:,1])

#Form a dataframe to operate on
df = pd.DataFrame(X)
mu,var = find_mean_and_variance(df)

p = Gaussian(X,mu,var)
pval = Gaussian(Xval,mu,var)

select_threshold(pval,yval)

