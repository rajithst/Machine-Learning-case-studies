# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:48:33 2019

@author: rajith
"""
%reset -f

#importing libraries and dataset from sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#create dataframe
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))


#sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean area','mean perimeter','mean smoothness'])
#sns.countplot(df_cancer['target'])
#sns.scatterplot(x="mean area",y="mean smoothness" , hue='target',data=df_cancer)
#sns.heatmap(df_cancer.corr(),annot=True)

#training the model
X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']

#split data to training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#create svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

classifier = SVC()
classifier.fit(X_train,y_train)

#predicting
y_pred = classifier.predict(X_test)

#confusion metrix and heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

#improve the model
#feature scaling 
# (X-Xmin)/Xmax- Xmin
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train-min_train)/range_train

#or using datapreprocessor
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#plot scatterolot
sns.scatterplot(x=X_train_scaled['mean area'],y=X_train_scaled['mean smoothness'],hue=y_train)

#scaling test set
# (X-Xmin)/Xmax- Xmin
min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test-min_test)/range_test

#train model again
classifier.fit(X_train_scaled,y_train)

#predicting again
y_pred = classifier.predict(X_test_scaled)

#confusion metrix and heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_pred))

#grid search
from sklearn.model_selection import GridSearchCV

param_grid = { 'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
grid = GridSearchCV(SVC(),param_grid,cv=3)
grid.fit(X_train_scaled,y_train)

grid.best_params_
grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test,grid_predictions)
sns.heatmap(cm,annot=True)
















