import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import Normalizer,MinMaxScaler
from matplotlib import pyplot as plt


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataset = pd.read_csv(url,delim_whitespace=True,names=names)

dataset = dataset.drop(labels=[ 'INDUS', 'NOX', 'TAX', 'B'],axis=1)


array = dataset.values

X = array[:,0:3]
y = array[:,3]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=3)

normalizer = MinMaxScaler().fit(X_train)

X_train_normalized = normalizer.transform(X_train)
X_test_normalized = normalizer.transform(X_test)

'''models = []
models.append(("Ridge",Ridge()))
models.append(("KNN",KNeighborsRegressor()))
models.append(("SVR",SVR()))
models.append(("AdaBoostRegressor",AdaBoostRegressor()))
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=13)
    cross_val = model_selection.cross_val_score(model,X_train_normalized,y_train,cv=kfold,scoring='r2')
    print("%s %0.2f,%0.2f"%(name,cross_val.mean(),cross_val.std()))'''


svr_model = GradientBoostingRegressor()
svr_model.fit(X_train_normalized,y_train)

y_pred = svr_model.predict(X_test_normalized)

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
