# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:33:35 2018

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
    df = pd.read_csv("ex1data1.csv",names=['Population','Profit'])
    df = df.sample(frac=1).reset_index(drop=True)
    
    #Printing the head of Dataframe
    '''print(df.head())
    
    print(df.info())'''
        
    ###Adding a column of ones to the dataframe
    X = pd.DataFrame(np.ones(len(df)))
    frames = [X,df['Population']]
    X = pd.concat(frames,axis=1)
    y = df['Profit']
    
    #Splitting Data set
    X_train = X.iloc[0:20,:]
    y_train = y[0:20]
    X_val = X.iloc[20:31,:]
    y_val = y[20:31] 
    X_test = X.iloc[31:41,:]
    y_test = y[31:41]
    
    '''  #Display a scatterplot
    plt.scatter(X_test['Population'],y_test,c='red',marker='x')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.xticks(range(4,26,2))
    plt.yticks(range(-5,30,5))'''
    return(X_train,y_train)
    
    
def Gradient_Descent(iteration=1500,alpha=0.01):    
    theta=np.ones((1,2))
    J_history = np.zeros((iteration,1))
    X_train,y_train = prepare_data()  
    m = len(X_train)
    

    for i in range(0,iteration):
        
        J,gradient =ComputeCost(theta,X_train,y_train,m)
        #Calculating Gradient to update theta values
        #gradient = (np.dot(loss,X_train))/m
        #Updation of theta values
        theta = theta-(alpha*(gradient))
        J_history[i] = J
        if i ==(iteration-1):
            print(J)
    return theta
    
    
def ComputeCost(theta,X,y,m,l=0.1):
    #Cost
    J = 0
    #Calculating the cost
    h = (np.matmul(theta,X.transpose()))
    h = h.reshape((len(y),))
    #Loss used to change theta
    loss=(h-y)
    #Loss square to measure the performance of gradient descent
    loss_squared = loss**2
    #Sum of squared loss
    sum_val = np.sum(loss_squared)
    #Cost
    J = (sum_val)/(2*m)
    J = round(J,2)
    #Regularization
    theta_except_bias = theta[1:,0]
    theta_squared_sum = np.sum((theta_except_bias)**2)
    regularization = (l/2*m)*theta_squared_sum
    #Regularized Cost
    J = J+regularization
    
     #Partial derivative of Cost with respect to theta
    costEffect = np.matmul((h-y),X)
    Gradient = (1/m)*((costEffect))
    Gradient_without_bias = Gradient[1:]
    #Regularization
    reg = (l/m)*theta_except_bias
    Gradient_without_bias = Gradient_without_bias+reg
    
    #Combining the bias term
    Gradient = np.insert(Gradient_without_bias,0,Gradient[0])
    
    return J, Gradient
    

def Square(n):
    return n**n
    
    
def main():
    theta = Gradient_Descent(100)
    #Optimal theta values
    print(theta)
main()