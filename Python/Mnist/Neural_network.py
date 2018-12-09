# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 23:10:59 2018

@author: Sandy
"""
import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score

#Implementation of Neural Network

#Preparing Data for mnist data
def prepare_data():
    data = scipy.io.loadmat("ex4data1.mat")
   
    X = data['X']
    y = data['y']
    return X,y

#Predefined weights
def predefined_weights():
    weights = scipy.io.loadmat("ex3weights.mat")
    
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    
    return Theta1,Theta2
    
    
#Initialize Random Weights
def randInitializeWeights(m,neurons=25,n=10):
    epsilon_init = 0.12
 
    Theta1 = np.random.rand(neurons,m[1])
    Theta1 = Theta1*(2*epsilon_init)-epsilon_init
    
    Theta2 = np.random.rand(n,neurons+1)
    Theta2 = Theta2*(2*epsilon_init)-epsilon_init
   
    return Theta1,Theta2

#Sigmoid function
def Sigmoid(X):
    #Exponential for sigmoid function
    exp = np.exp(-X)    
    #Denominator value
    denominator = 1+exp
    #Sigmoid value
    Sigmoid_value = (1/denominator)
    
    return Sigmoid_value

def Sigmoid_gradient(X):
    return((Sigmoid(X)*(1-Sigmoid(X))))

def predict(x,theta1,theta2):
    #Forward Propagation Block
    #Input layer
    a1 = x
  
    #Hidden Layer
    z2 = np.matmul(a1,theta1.transpose())
    a2 = Sigmoid(z2)
    #Adding bias term
    bias_column = np.ones((len(a2)))
    a2 = np.insert(a2,0,bias_column,axis=1)
    
    #Output Layer
    z3 = np.matmul(a2,theta2.transpose())   
    a3 = Sigmoid(z3)
    
    #Convert probabilities into predictions
    y_pred = np.zeros((len(x),1))
    
    for pos,each in enumerate(a3):
        val = np.argmax(each)
        y_pred[pos] = val
    #print(y_pred)
    return a3,y_pred

#Forward Propagation
def One_Pass(x,y,theta1,theta2):
    #Forward Propagation Block
    #Input layer
    a1 = x
    
    #Hidden Layer
    z2 = np.matmul(a1,theta1.transpose())
    a2 = Sigmoid(z2)
    
    #Adding bias term
    bias_column = np.ones((1,1))
    a2 = np.insert(a2,0,bias_column)
    
    #Output Layer
    z3 = np.matmul(a2,theta2.transpose())
    a3 = Sigmoid(z3)
    
    #Back Propagation block
    #Delta of output layer
    delta_3 = y-a3
    theta2 = theta2[:,1:]
   
    #Delta of hidden layer
    first_term = np.matmul(theta2.transpose(),delta_3)
    second_term = Sigmoid_gradient(z2)
    delta_2 = first_term*second_term
   
    a1 = a1[1:]
    a2 = a2[1:]
    
    a2 = a2.reshape((len(a2),1))
    delta_3 = delta_3.reshape((len(delta_3),1))
    
    a1 = a1.reshape((len(a1),1))
    delta_2 = delta_2.reshape((len(delta_2),1))

    #Sum of Delta
    Delta_2 = (np.matmul(a2,delta_3.transpose()))
    Delta_1 = (np.matmul(a1,delta_2.transpose()))

    return Delta_1,Delta_2


#One hot encoding
def oneHotEncoding(y,label_count):
    y_encoded = np.eye(label_count)[y]
    y_encoded = y_encoded.reshape((len(y),label_count))
    return y_encoded

#Compute the cost
def ComputeCost(h,y,theta1,theta2,m,l=0.1):
    #Cost
    Cost = 0
    
    #h = h.reshape(m,)
    #First term with respect to probability of y
    firstTerm = y*np.log(h)
    #Second term with respect to probability of 1-y
    secondTerm = (1-y)*(np.log(1-h))
    #Sum of both the probabilities
    logvalue = -(firstTerm)-(secondTerm)
    #Cost function
    Cost = (1/m)*(np.sum(logvalue))
    
    #Regularization
    theta1_except_bias = theta1[:,1:]
    theta2_except_bias = theta2[:,1:]
    
    theta1_squared_sum = np.sum(np.square(theta1_except_bias))
    theta2_squared_sum = np.sum(np.square(theta2_except_bias))
   
    numerator = l*theta1_squared_sum+theta2_squared_sum
    denominator = 2*m
    #Regularization term
    regularization = numerator/denominator
    #Regularized Cost
    Cost = Cost+regularization
    
    return Cost


def main():
    X,y = prepare_data()
    y[y==10] = 0
    
    bias_column = np.ones((len(X)))
    
    X = np.insert(X,0,bias_column,axis=1)
    
    y_encoded = oneHotEncoding(y,10)
    shape = X.shape
    m = shape[0]
    n = 10
    
    #theta1,theta2 = predefined_weights()
    #Randomly initialize weights
    theta1,theta2 = randInitializeWeights(shape,25,n)
   
    '''y_pred_encoded,y_pred = predict(X,theta1,theta2)
    #compute the cost of neural network
    Cost = ComputeCost(y_pred_encoded,y_encoded,theta1,theta2,m,1)
    print(Cost)  
    print(accuracy_score(y,y_pred)*100)'''
    
    for i in range(10):
        Delta_1 = 0
        Delta_2 = 0
        for x,y_ in zip(X,y_encoded):
           Del_1,Del_2 =  One_Pass(x,y_,theta1,theta2)
           Delta_1 = Delta_1+Del_1
           Delta_2 = Delta_2+Del_2
        
        theta1_without_bias = theta1[:,1:]
        theta2_without_bias = theta2[:,1:]
    
        multiplier = (0.1/m)
        #Neural network cost
        second_term = multiplier*theta1_without_bias
        Delta_1 = np.divide(Delta_1,m)+second_term.transpose()
        second_term = multiplier*theta2_without_bias
        Delta_2 = np.divide(Delta_2,m)+second_term.transpose()
        
        theta1_without_bias = theta1_without_bias-(0.01*Delta_1.transpose())
        theta2_without_bias = theta2_without_bias-(0.01*Delta_2.transpose())
        
        theta1 = np.insert(theta1_without_bias,0,theta1[:,0],axis=1)
        theta2 = np.insert(theta2_without_bias,0,theta2[:,0],axis=1)
            
        y_pred_encoded,y_pred = predict(X,theta1,theta2)
        #compute the cost of neural network
        Cost = ComputeCost(y_pred_encoded,y_encoded,theta1,theta2,m,1)
        
        print(Cost)  
        print(y)
        print(y_pred)
        print(accuracy_score(y,y_pred)*100)
        

main()
  
    
    
    
    
    
    
    
    
    
    
    
    
    