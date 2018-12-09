# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:59:31 2018

@author: Sandy
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

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
    df = pd.read_csv("ex2data1.csv",names=['Exam1','Exam2','Admission'])
    X = df.drop('Admission',axis=1)
    X = Normalize(X)
    y = df['Admission']
    
    '''C = ['Red','Blue']
    -
    plt.legend(['Admitted','Not Admitted'])
    plt.show()'''
 
    
    ones = pd.DataFrame(np.ones(len(df)))
    frames = [ones,X['Exam1'],X['Exam2']]
    X = pd.concat(frames,axis=1)
    
    return(X,y)

#Sigmoid function
def Sigmoid(X):
    #Exponential for sigmoid function
    exp = np.exp(-X)    
    #Denominator value
    denominator = 1+exp
    #Sigmoid value
    Sigmoid_value = (1/denominator)
    
    return Sigmoid_value
    
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
    
 
def Gradient_Descent(iteration=1500,alpha=1):    
    X,y = prepare_data() 
    
    m = len(X)
    n = len(X.columns)
    
    theta = np.zeros((1,n))
    J_history = np.zeros((iteration,1))
    
    for i in range(0,iteration):
        Cost,Gradient = ComputeCost(theta,X,y)
        theta = theta - (alpha*Gradient)/m
        if i == (iteration-1):
            print(Cost)
        J_history[i]=Cost
    ''' fig,ax=plt.subplots()
    ax.plot(np.arange(iteration),J_history,'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Plot')'''
    
    return theta
    
    
def ComputeCost(theta,X,y):
    #Cost
    Cost = 0
    
    m = len(X)
    n = len(X.columns)
    
    z = np.matmul(theta,X.transpose())
    h = Sigmoid(z)
    h = h.reshape(100,)
    #First term with respect to probability of y
    firstTerm = np.dot(y,np.log(h))
    #Second term with respect to probability of 1-y
    secondTerm = np.dot((1-y),(np.log(1-h)))
    #Sum of both the probabilities
    logvalue = -(firstTerm)-(secondTerm)
    #Cost function
    Cost = (1/m)*(np.sum(logvalue))
    #Partial derivative of Cost with respect to theta
    costEffect = np.matmul((h-y),X)
    Gradient = (1/m)*((costEffect))
    return Cost, Gradient

def Threshold(Value=0.5):
    if Value>=0.5:
        return 1
    else:
        return 0
    
def Predict(theta,X):
    z = np.matmul(theta,X.transpose())
    predict_proba = Sigmoid(z)
    
    threshold_func = np.vectorize(Threshold)
    predictions = threshold_func(predict_proba)
    return predict_proba,predictions

def Square(n):
    return n**n
    
    
def main():
    s = Gradient_Descent(1000)
    X,y = prepare_data()
    
    predict_proba,y_pred = Predict(s,X)
    y_pred = y_pred.reshape(len(X),)
    
    ##Scatter plot with Original labels
    plt.subplot(2,1,1)
    sns.scatterplot(X['Exam1'],X['Exam2'],hue=y)
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")
    #Scatter plot with predicted labels
    plt.subplot(2,1,2)
    sns.scatterplot(X['Exam1'],X['Exam2'],hue=y_pred)
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")
    #Accuracy
    print("Accuracy:%s"%accuracy_score(y,y_pred))


main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    