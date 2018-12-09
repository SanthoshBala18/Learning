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
    
    #Printing the head of Dataframe
    '''print(df.head())
    
    print(df.info())'''
    
    #Display a scatterplot
    '''plt.scatter(df['Population'],df['Profit'],c='red',marker='x')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.xticks(range(4,26,2))
    plt.yticks(range(-5,30,5))'''
    
    ###Adding a column of ones to the dataframe
    X = pd.DataFrame(np.ones(len(df)))
    frames = [X,df['Population']]
    X = pd.concat(frames,axis=1)
    y = df['Profit']
    return(X,y)
    
    
def Gradient_Descent(iteration=1500,alpha=0.01):    
    theta = np.array((-1,2))
    J_history = np.zeros((iteration,1))
    X,y = prepare_data()  
    m = len(X)
    
    for i in range(0,iteration):
        print(theta)
        J,loss = ComputeCost(theta,X,y,m)
        #Calculating Gradient to update theta values
        gradient = (np.dot(loss,X))/m
        #Updation of theta values
        theta = theta-(alpha*gradient)
        J_history[i] = J
    return theta
    
    
def ComputeCost(theta,X,y,m):
    #Cost
    J = 0
    #Calculating the cost
    h = (np.matmul(theta.transpose(),X.transpose()))
    h = h.reshape((97,))
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
    theta = Gradient_Descent(1200)
    #Optimal theta values
    print(theta)
    
main()
    