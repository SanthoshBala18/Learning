# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 23:51:06 2018

@author: Sandy
"""

import scipy.io
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

mat = scipy.io.loadmat("ex6data1")
#Reading input variables
X = mat['X']
X = pd.DataFrame(X,columns=['X1','X2'])
#Reading output variables
y = mat['y']
y = y.reshape(51,)
#Plotting input variable and output variable
sns.scatterplot('X1','X2',data=X,hue=y,style=y)

#SVC model
model = SVC()
model.fit(X,y)

y_pred = model.predict(X)

print(accuracy_score(y_pred,y))