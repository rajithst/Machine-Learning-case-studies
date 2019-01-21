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

#visualize the data
#all plots for view correlation
sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean area','mean perimeter','mean smoothness'])

#count the plot
sns.countplot(df_cancer['target'])

#scatterplot
sns.scatterplot(x="mean area",y="mean smoothness" , hue='target',data=df_cancer)

#show heatmap
sns.heatmap(df_cancer.corr(),annot=True)

#training the model
X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']

#split data to training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

classifier = SVC()
classifier.fit(X_train,y_train)

#evaluating model
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)