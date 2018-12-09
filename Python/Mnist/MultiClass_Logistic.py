# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:44:29 2018

@author: Sandy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def Visualize(df):
    label = df.iloc[0,0]
    pixels = df.iloc[0,1:]
    
    pixels = np.array(pixels,dtype='uint8')
    
    pixels = pixels.reshape((28,28))
    
    plt.imshow(pixels,cmap='gray')
    plt.show()
    

def prepare_data():
    
    df = pd.read_csv("mnist_train.csv")
    X = df.drop('label',axis=1)
    y = df['label']
    
    ones = np.ones(len(df))
    X.insert(0,'bias',ones)
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
    
def Gradient_Descent(X,y,iteration=1500,alpha=0.01,l=0.1):    
 
    m = len(X)
    n = len(X.columns)
    
    theta = np.zeros((1,n))
    J_history = np.zeros((iteration,1))
    
    for i in range(0,iteration):
        Cost,Gradient = ComputeCost(theta,X,y,l)
        theta = theta - (alpha*Gradient)/m
       
        J_history[i]=Cost
    ''' fig,ax=plt.subplots()
    ax.plot(np.arange(iteration),J_history,'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Plot')'''
    
    return theta
    
    
def ComputeCost(theta,X,y,l):
    #Cost
    Cost = 0
    
    m = len(X)
    n = len(X.columns)
    
    z = np.matmul(theta,X.transpose())
    h = Sigmoid(z)
    h = h.reshape(m,)
    #First term with respect to probability of y
    firstTerm = np.dot(y,np.log(h))
    #Second term with respect to probability of 1-y
    secondTerm = np.dot((1-y),(np.log(1-h)))
    #Sum of both the probabilities
    logvalue = -(firstTerm)-(secondTerm)
    #Cost function
    Cost = (1/m)*(np.sum(logvalue))
    #Regularization
    theta_except_bias = theta[0,1:]
    theta_squared_sum = np.sum((theta_except_bias)**2)
    regularization = (l/2*m)*theta_squared_sum
    #Regularized Cost
    Cost = Cost+regularization
    #Partial derivative of Cost with respect to theta
    costEffect = np.matmul((h-y),X)
    Gradient = (1/m)*((costEffect))
    Gradient_without_bias = Gradient[1:]
    #Regularization
    reg = (l/m)*theta_except_bias
    Gradient_without_bias = Gradient_without_bias+reg
    #Combining the bias term
    Gradient = np.insert(Gradient_without_bias,0,Gradient[0],axis=0)
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
    
def Convert_y_to_binary(val,i):
    if val == i:
        return 1
    else:
        return 0
    
def main():
    
    X,y = prepare_data()
    K = 10
    conversion = np.vectorize(Convert_y_to_binary)
    theta = np.zeros((len(X.columns),K))
    for each in range(0,K):
        bin_y = conversion(each,y)
        theta[each,:] = Gradient_Descent(X,bin_y,20)
    
    print(theta)
        
    '''X,y = prepare_data()
    
    predict_proba,y_pred = Predict(s,X)
    y_pred = y_pred.reshape(len(X),)
    print(predict_proba)
    
    sns.scatterplot(X['Exam1'],X['Exam2'],hue=y)
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")'''
    '''theta = Gradient_Descent(1000)
    #Optimal theta values
    print(theta)'''
    
main()


    
    
    
    
    