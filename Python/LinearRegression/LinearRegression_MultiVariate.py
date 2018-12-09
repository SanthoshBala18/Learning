# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:59:31 2018

@author: Sandy
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data():
    '''#Reading data from the file
    with open("ex1data1.txt") as f:
        data = f.read().strip()
    
    population = []
    profit = []
    
    #Splitting the columns of data
    for each in data.split("\n"):
        values = each.split(',')
        population.append(values[0])
        profit.append(values[1])'''
    
    #Creating the dataframe
    df = pd.read_csv("ex1data2.csv",names=['Size','Bedrooms','Price'])
    
    X = Normalize(df)
    y=df['Price'].values
    #Printing the head of Dataframe
    print(df.head())
    
    print(df.info())
    
    ones = pd.DataFrame(np.ones(len(df)))
    frames = [ones,X['Size'],X['Bedrooms']]
    X = pd.concat(frames,axis=1)
    
    
    return(X,y)

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
    
 
def Gradient_Descent(iteration=1500,alpha=0.01):    
    X,y = prepare_data() 
    
    m = len(X)
    n = len(X.columns)
    
    theta = np.zeros((1,n))
    J_history = np.zeros((iteration,1))
    
    for i in range(0,iteration):
        J,loss = ComputeCost(theta,X,y)
        #theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)

        #Calculating Gradient to update theta values
        gradient = np.matmul(loss.transpose(),X)

        #Updation of theta values
        theta = theta-(alpha*gradient/m)
        J_history[i] = J
        
        if i == (iteration-1):
            print(J)
    fig,ax=plt.subplots()
    ax.plot(np.arange(iteration),J_history,'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Plot')
    
    return theta
    
    
def ComputeCost(theta,X,y):
    #Cost
    J = 0
    '''tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))'''
    
    m = len(X)
    n = len(X.columns)
    
    #Calculating the cost
    h = (np.matmul(X,theta.transpose()))
    h = h.reshape((m,))
    #Loss used to change theta
    loss=(h-y)
    #Loss square to measure the performance of gradient descent
    loss_squared = loss**2
    #Sum of squared loss
    sum_val = np.sum(loss_squared)
    
    #Cost
    J = (sum_val)/(2*m)
    J = round(J,2)
    return J,loss
    

def Square(n):
    return n**n
    
    
def main():
    theta = Gradient_Descent(1000)
    #Optimal theta values
    print(theta)


main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    