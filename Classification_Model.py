import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


dataset_url = 'Cryotherapy.csv'
data = pd.read_csv(dataset_url,sep=",",index_col=0)
'''data = data.drop(['Number_of_Warts'],axis=1)

data.loc[ data['age'] <= 26, 'age'] = 0
data.loc[(data['age'] > 26) & (data['age'] <= 36), 'age'] = 1
data.loc[(data['age'] > 36) & (data['age'] <= 46), 'age'] = 2
data.loc[(data['age'] > 46) & (data['age'] <= 56), 'age'] = 3
data.loc[ data['age'] > 56, 'age'] = 4

data.loc[ data['Area'] <= 150, 'Area'] = 0
data.loc[(data['Area'] > 150) & (data['Area'] <= 310), 'Area'] = 1
data.loc[(data['Area'] > 310), 'Area'] = 2'''

training_result = data['Result_of_Treatment']
training_data = data.drop(['Result_of_Treatment'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(training_data,training_result,test_size=.30,random_state=123)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

acc_random_forest = round(model.score(X_train, y_train) * 100, 2)
print(acc_random_forest)
print(accuracy_score(y_test,y_pred))
print(y_test,y_pred)

