from sklearn import datasets
import pandas
from pandas import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report,log_loss
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import numpy as np
from sklearn.datasets import load_iris
# Load dataset


dataset = load_iris()
X,Y = dataset['data'],dataset['target']


'''url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#dataset['class'] = dataset['class'].astype('category').cat.codes
array = dataset.values
X = array[:,0:4]
Y = array[:,4]'''

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size = 0.20,random_state=123)

'''BinaryTransformer = preprocessing.Binarizer(threshold=0.5).fit(X_train)
print(BinaryTransformer.transform(X_train))'''
#print(BinaryTransformer[:,0])


'''Quantile_Transformer = preprocessing.QuantileTransformer(random_state=0)
X_transformed = Quantile_Transformer.fit_transform(X_train)
print(X_transformed)
print(np.percentile(X_transformed[:,0],[0,25,50,75,100]))'''


'''dataset = preprocessing.scale(dataset)

print(dataset.mean(axis=0))
print(dataset.std(axis=0))'''
'''scaler = StandardScaler().fit(dataset)
dataset = scaler.transform(dataset)

print(dataset.mean(axis=0))
print(dataset.std(axis=0))'''



'''print(dataset['class'])
print(dataset.corr())'''
'''array = dataset.values
X = array[:,0:4]
Y = array[:,4]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size = 0.20,random_state=123)'''


'''models = []

models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('SVC',SVC()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('GNB',GaussianNB()))'''

models = []
models.append(('Forrest',RandomForestClassifier()))
models.append(('Gradient',GradientBoostingClassifier()))
models.append(('AdaGradient',AdaBoostClassifier()))

results= [ ]
names=[]
for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=123)
    cv_results = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s,%f,%f"%(name,cv_results.mean(),cv_results.std()))

'''LDA = LinearDiscriminantAnalysis()

LDA.fit(X_train,y_train)

predict_value = LDA.predict(X_test)
predict_prob = LDA.predict_proba(X_test)
print(predict_value)
print(y_test)

print(accuracy_score(y_test,predict_value))
print(confusion_matrix(y_test,predict_value))
print(classification_report(y_test,predict_value))
print(log_loss(y_test,predict_prob))
'''